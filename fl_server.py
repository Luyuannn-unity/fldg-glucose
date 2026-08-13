"""
Real (networked) FL server for CGMTransformer — you are the server.

Holds the authoritative global model, coordinates synchronous FedAvg rounds
over HTTP, and never sees any client's data. Each round is two phases:

  TRAIN  clients fetch the current global, train one local round, POST their
         weights + sample count; once all `num_clients` submit, the server
         FedAvg-aggregates them into the next global.
  EVAL   clients fetch the just-aggregated global, evaluate it on their own
         val split, POST val MSE/RMSE; the server averages these, tracks the
         best round, and applies early stopping.

After the last round (or early stop) a TEST phase evaluates the best global on
each client's test split. The server writes per-round metrics, per-client test
metrics, and best_global_model.pt to <out_dir>.

Clients are driven entirely by GET /task, so they can live on any machine that
can reach this server's host:port.

Run:
  python fl_server.py --config config.yaml --port 8088 --num-clients 4
"""

import argparse
import copy
import csv
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from omegaconf import OmegaConf

import fl_common as C


class FLState:
    """All shared FL state, guarded by a single lock. The HTTP handler threads
    only touch state through these methods."""

    def __init__(self, cfg, num_clients, num_rounds, patience, out_dir):
        self.lock = threading.Lock()
        self.num_clients = int(num_clients)
        self.num_rounds = int(num_rounds)
        self.patience = int(patience)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # phase: "train" -> "eval" -> (next round) ... -> "test" -> "done"
        self.phase = "train"
        self.round = 1

        # Models held in memory: just the three we ever serve.
        print("Building initial global model (server-side, no data)...", flush=True)
        self.current_model = C.build_initial_state_bytes(cfg)   # global for self.round
        self.pending_model = None                               # aggregate awaiting eval
        self.best_blob = None
        print(f"  initial model: {len(self.current_model)/1e6:.1f} MB", flush=True)

        # Per-phase submissions for the current round.
        self.train_subs = {}   # client_id -> {"blob","n_train","train_loss"}
        self.eval_subs = {}     # client_id -> {"val_mse","val_rmse"}
        self.test_subs = {}     # client_id -> metrics dict

        self.registered = {}    # client_id -> {"source_name","n_train"}

        # Best-round tracking (on the aggregated model's avg val MSE).
        self.best_val = float("inf")
        self.best_round = None
        self.rounds_since_best = 0

        # History for CSVs.
        self.train_loss_by_round = {}   # round -> {source_name: loss}
        self.val_by_round = {}          # round -> {source_name: (val_mse, val_rmse)}
        self.summary_rows = []          # (round, avg_val_mse, avg_val_rmse, is_best)
        # PFL clients poll GET /round_info?round=r to learn whether the
        # just-completed round was the new global best (so they can snapshot
        # their personal/mixture deploy blob locally). Populated in
        # _finish_round_locked.
        self.round_best_flags = {}      # round -> bool

    # ── registration ────────────────────────────────────────────────────────
    def register(self, client_id, source_name, n_train):
        evicted = []
        with self.lock:
            # If someone re-registers the same cohort under a different
            # client_id (e.g. they Ctrl+C'd and relaunched the wizard, typing
            # a different Client ID this time), evict the phantom(s) so the
            # server doesn't wait forever for a client that no longer exists.
            for old_id, info in list(self.registered.items()):
                if info["source_name"] == source_name and old_id != client_id:
                    del self.registered[old_id]
                    self.train_subs.pop(old_id, None)
                    self.eval_subs.pop(old_id, None)
                    self.test_subs.pop(old_id, None)
                    evicted.append(old_id)
            self.registered[client_id] = {"source_name": source_name, "n_train": n_train}
            n = len(self.registered)
        for old_id in evicted:
            print(f"[register] evicted phantom '{old_id}' (same cohort={source_name})", flush=True)
        print(f"[register] {client_id}  cohort={source_name}  n_train={n_train}  "
              f"({n}/{self.num_clients})", flush=True)
        return {"ok": True, "num_rounds": self.num_rounds, "num_clients": self.num_clients}

    def _source_of(self, client_id):
        return self.registered.get(client_id, {}).get("source_name", client_id)

    # ── task dispatch ─────────────────────────────────────────────────────────
    def get_task(self, client_id):
        with self.lock:
            if self.phase == "done":
                return {"action": "done"}
            if self.phase == "train":
                if client_id in self.train_subs:
                    return {"action": "wait"}
                return {"action": "train", "round": self.round, "model_tag": "current"}
            if self.phase == "eval":
                if client_id in self.eval_subs:
                    return {"action": "wait"}
                return {"action": "eval", "round": self.round, "model_tag": "pending"}
            if self.phase == "test":
                if client_id in self.test_subs:
                    return {"action": "wait"}
                return {"action": "test", "round": self.round, "model_tag": "best"}
            return {"action": "wait"}

    def get_model(self, tag):
        with self.lock:
            if tag == "current":
                return self.current_model
            if tag == "pending":
                return self.pending_model
            if tag == "best":
                return self.best_blob if self.best_blob is not None else self.current_model
        return None

    # ── submissions ───────────────────────────────────────────────────────────
    def submit_train(self, client_id, rnd, n_train, train_loss, blob):
        with self.lock:
            if self.phase != "train" or rnd != self.round:
                return {"ok": False, "msg": "stale"}
            self.train_subs[client_id] = {
                "blob": blob, "n_train": int(n_train), "train_loss": float(train_loss),
            }
            src = self._source_of(client_id)
            self.train_loss_by_round.setdefault(self.round, {})[src] = float(train_loss)
            got, need = len(self.train_subs), self.num_clients
            print(f"[round {self.round}] train update from {src} "
                  f"(train_loss={train_loss:.5f}, n={n_train})  [{got}/{need}]", flush=True)
            if got >= need:
                self._aggregate_locked()
            return {"ok": True}

    def _aggregate_locked(self):
        blobs = [s["blob"] for s in self.train_subs.values()]
        counts = [s["n_train"] for s in self.train_subs.values()]
        self.pending_model = C.fedavg(blobs, counts)
        print(f"[round {self.round}] aggregated {len(blobs)} clients (FedAvg) "
              f"-> {len(self.pending_model)/1e6:.1f} MB. entering EVAL.", flush=True)
        self.train_subs = {}
        self.phase = "eval"

    def submit_eval(self, client_id, rnd, val_mse, val_rmse):
        with self.lock:
            if self.phase != "eval" or rnd != self.round:
                return {"ok": False, "msg": "stale"}
            self.eval_subs[client_id] = {"val_mse": float(val_mse), "val_rmse": float(val_rmse)}
            src = self._source_of(client_id)
            got, need = len(self.eval_subs), self.num_clients
            print(f"[round {self.round}] eval from {src}: "
                  f"val_mse={val_mse:.6f} val_rmse={val_rmse:.3f}  [{got}/{need}]", flush=True)
            if got >= need:
                self._finish_round_locked()
            return {"ok": True}

    def get_round_info(self, rnd):
        """PFL clients call this after eval-submit to learn whether their
        just-submitted round was the new best. Returns complete=False while any
        client still hasn't submitted; complete=True + is_best once the round
        has been aggregated."""
        with self.lock:
            hit = self.round_best_flags.get(int(rnd))
            if hit is None:
                return {"complete": False}
            return {"complete": True, "is_best": bool(hit)}

    def _finish_round_locked(self):
        mses = [v["val_mse"] for v in self.eval_subs.values() if math.isfinite(v["val_mse"])]
        rmses = [v["val_rmse"] for v in self.eval_subs.values() if math.isfinite(v["val_rmse"])]
        avg_mse = float(sum(mses) / len(mses)) if mses else float("inf")
        avg_rmse = float(sum(rmses) / len(rmses)) if rmses else float("inf")

        # Persist per-client val before eval_subs is cleared.
        self.val_by_round[self.round] = {
            self._source_of(cid): (v["val_mse"], v["val_rmse"])
            for cid, v in self.eval_subs.items()
        }

        improved = avg_mse < self.best_val
        # PFL clients read this via /round_info to know when to snapshot their
        # personal/mixture deploy blob.
        self.round_best_flags[self.round] = bool(improved)
        if improved:
            self.best_val = avg_mse
            self.best_round = self.round
            self.best_blob = self.pending_model
            self.rounds_since_best = 0
        else:
            self.rounds_since_best += 1

        self.summary_rows.append((self.round, avg_mse, avg_rmse, improved))
        flag = "  <-- new best" if improved else ""
        print(f"[round {self.round}] avg val MSE={avg_mse:.6f}  avg val RMSE(30m)="
              f"{avg_rmse:.3f} mg/dL{flag}", flush=True)

        # The aggregated model becomes the global for the next round.
        self.current_model = self.pending_model
        self.pending_model = None
        self.eval_subs = {}

        self._save_round_metrics_locked()

        stop = (self.round >= self.num_rounds) or \
               (self.patience > 0 and self.rounds_since_best >= self.patience)
        if stop:
            if self.best_blob is None:
                self.best_blob = self.current_model
                self.best_round = self.round
            self._save_best_model_locked()
            reason = ("reached num_rounds" if self.round >= self.num_rounds
                      else f"early stop (no val improvement for {self.patience} rounds)")
            print(f"[server] training finished: {reason}. "
                  f"best round={self.best_round} (avg val MSE={self.best_val:.6f}). "
                  f"entering TEST on best model.", flush=True)
            self.phase = "test"
        else:
            self.round += 1
            self.phase = "train"
            print(f"[server] --- advancing to round {self.round} (TRAIN) ---", flush=True)

    def submit_test(self, client_id, metrics):
        with self.lock:
            if self.phase != "test":
                return {"ok": False, "msg": "stale"}
            src = self._source_of(client_id)
            self.test_subs[client_id] = {"source_name": src, "metrics": metrics}
            got, need = len(self.test_subs), self.num_clients
            r30 = metrics.get("rmse_mgdl_30min", float("nan"))
            print(f"[test] {src}: rmse_30={r30}  [{got}/{need}]", flush=True)
            if got >= need:
                self._save_test_metrics_locked()
                self.phase = "done"
                print("[server] ===== FEDERATED LEARNING COMPLETE =====", flush=True)
                print(f"[server] best round={self.best_round}, avg val MSE={self.best_val:.6f}",
                      flush=True)
                print(f"[server] outputs in: {os.path.abspath(self.out_dir)}", flush=True)
            return {"ok": True}

    # ── output files ──────────────────────────────────────────────────────────
    def _save_round_metrics_locked(self):
        # Summary: one row per round.
        with open(os.path.join(self.out_dir, "round_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "avg_val_mse", "avg_val_rmse_mgdl_30min", "new_best"])
            for r, mse, rmse, best in self.summary_rows:
                w.writerow([r, mse, rmse, int(best)])
        # Long: one row per (round, client) with train loss + val metrics.
        with open(os.path.join(self.out_dir, "round_client_metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["round", "client", "train_loss", "val_mse", "val_rmse_mgdl_30min"])
            for r in sorted({r for r, _, _, _ in self.summary_rows}):
                tl = self.train_loss_by_round.get(r, {})
                vl = self.val_by_round.get(r, {})
                for src in sorted(set(tl) | set(vl)):
                    vm, vr = vl.get(src, ("", ""))
                    w.writerow([r, src, tl.get(src, ""), vm, vr])

    def _save_best_model_locked(self):
        path = os.path.join(self.out_dir, "best_global_model.pt")
        with open(path, "wb") as f:
            f.write(self.best_blob)
        with open(os.path.join(self.out_dir, "best_global_model_meta.txt"), "w") as f:
            f.write(f"best_round={self.best_round}\n")
            f.write(f"best_avg_val_mse={self.best_val}\n")
            f.write(f"num_rounds={self.num_rounds}\n")
        print(f"[server] saved best global model -> {path}", flush=True)

    def _save_test_metrics_locked(self):
        cols = ["mse_norm", "mse_mgdl_30min", "rmse_mgdl_30min",
                "tir_actual", "tir_predicted", "tir_difference"]
        path = os.path.join(self.out_dir, "best_model_test_metrics.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["client"] + cols)
            for sub in sorted(self.test_subs.values(), key=lambda s: s["source_name"]):
                m = sub["metrics"]
                w.writerow([sub["source_name"]] + [m.get(c, "") for c in cols])
        print(f"[server] saved per-client test metrics -> {path}", flush=True)

    def status(self):
        with self.lock:
            return {
                "phase": self.phase,
                "round": self.round,
                "num_rounds": self.num_rounds,
                "num_clients": self.num_clients,
                "registered": len(self.registered),
                "train_subs": len(self.train_subs),
                "eval_subs": len(self.eval_subs),
                "test_subs": len(self.test_subs),
                "best_round": self.best_round,
                "best_val_mse": (None if self.best_val == float("inf") else self.best_val),
            }


# Global state, set in main(); handler threads read it.
STATE: FLState = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # we do our own logging in FLState

    def _send_json(self, obj, code=200):
        import json
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data, code=200):
        if data is None:
            return self._send_json({"ok": False, "msg": "model not available"}, code=503)
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(n) if n else b""

    def _qs(self, parsed):
        return {k: v[0] for k, v in parse_qs(parsed.query).items()}

    def do_GET(self):
        parsed = urlparse(self.path)
        q = self._qs(parsed)
        if parsed.path == "/task":
            return self._send_json(STATE.get_task(q.get("client_id", "?")))
        if parsed.path == "/round_info":
            try:
                r = int(q.get("round", "0"))
            except ValueError:
                return self._send_json({"complete": False, "msg": "bad round"}, code=400)
            return self._send_json(STATE.get_round_info(r))
        if parsed.path == "/model":
            return self._send_bytes(STATE.get_model(q.get("tag", "current")))
        if parsed.path == "/status":
            return self._send_json(STATE.status())
        return self._send_json({"ok": False, "msg": "not found"}, code=404)

    def do_POST(self):
        import json
        parsed = urlparse(self.path)
        q = self._qs(parsed)
        if parsed.path == "/register":
            body = json.loads(self._read_body().decode())
            return self._send_json(STATE.register(
                body["client_id"], body.get("source_name", "?"), int(body.get("n_train", 0))))
        if parsed.path == "/train_result":
            blob = self._read_body()
            return self._send_json(STATE.submit_train(
                q["client_id"], int(q["round"]), int(q.get("n_train", 0)),
                float(q.get("train_loss", "nan")), blob))
        if parsed.path == "/eval_result":
            body = json.loads(self._read_body().decode())
            return self._send_json(STATE.submit_eval(
                q["client_id"], int(q["round"]),
                float(body.get("val_mse", "inf")), float(body.get("val_rmse", "inf"))))
        if parsed.path == "/test_result":
            body = json.loads(self._read_body().decode())
            return self._send_json(STATE.submit_test(q["client_id"], body.get("metrics", {})))
        return self._send_json({"ok": False, "msg": "not found"}, code=404)


def main():
    ap = argparse.ArgumentParser(description="CGMTransformer real-FL server")
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--num-clients", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    s = cfg.get("server", {})
    host = a.host or s.get("host", "0.0.0.0")
    port = a.port or int(s.get("port", 8088))
    num_clients = a.num_clients or int(s.get("num_clients", 4))
    rounds = a.rounds or int(s.get("num_rounds", 25))
    patience = a.patience if a.patience is not None else int(s.get("early_stop_patience", 5))
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.config)),
                                         s.get("out_dir", "output"))

    global STATE
    STATE = FLState(cfg, num_clients=num_clients, num_rounds=rounds,
                    patience=patience, out_dir=out_dir)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print("=" * 70, flush=True)
    print(f"CGMTransformer real-FL server listening on http://{host}:{port}", flush=True)
    print(f"  clients expected : {num_clients}", flush=True)
    print(f"  rounds           : {rounds}  (early-stop patience={patience})", flush=True)
    print(f"  out dir          : {os.path.abspath(out_dir)}", flush=True)
    print(f"  method           : {cfg.train_args.federated_optimizer_name}"
          f"{' + MLDG' if cfg.train_args.get('mldg', False) else ''}", flush=True)
    print("  waiting for clients to register...", flush=True)
    print("=" * 70, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] interrupted, shutting down.", flush=True)
        httpd.shutdown()


if __name__ == "__main__":
    main()
