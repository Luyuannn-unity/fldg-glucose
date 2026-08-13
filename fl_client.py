"""
Real (networked) FL client for CGMTransformer — one per cohort.

Wraps the existing FLockModelTransformer (model + data + train/eval, MLDG/FedProx
all included) and drives it from the server's GET /task state machine:

  task "train" -> train one local round on the global weights, POST the update
  task "eval"  -> evaluate the aggregated global on this cohort's val split, POST
  task "test"  -> evaluate the best global on this cohort's test split, POST
  task "wait"  -> other clients still working this phase; sleep and re-poll
  task "done"  -> federation finished; exit

The cohort's data never leaves this process — only model weights and scalar
metrics are sent to the server.

Run:
  python fl_client.py --config config.yaml \
      --source ./data_output/metabonet_splits/HUPA-UCM \
      --server http://SERVER_HOST:8088
"""

import argparse
import os
import socket
import time

from omegaconf import OmegaConf

import fl_common as C


def seed_everything(seed: int):
    """Reproducibility setup (RNGs + cuDNN determinism + PYTHONHASHSEED),
    matching the original simulation so a real-FL run is reproducible against
    itself."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_client_args(cfg, source_dir, out_root):
    """Materialise an OmegaConf args object FLockModelTransformer expects, pointing
    at this client's single cohort dir."""
    args = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_struct(args, False)
    args.data_args.source_dir = source_dir          # singular: this client's data
    args.tracking_args.out_put_root = out_root
    return args


def main():
    ap = argparse.ArgumentParser(description="CGMTransformer real-FL client")
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True,
                    help="this client's cohort split dir (its private data)")
    ap.add_argument("--server", default="http://127.0.0.1:8088")
    ap.add_argument("--client-id", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="where this client writes logs/plots (default: <config>/output/clients/<cohort>)")
    ap.add_argument("--poll", type=float, default=3.0, help="seconds between /task polls when waiting")
    a = ap.parse_args()

    from transformer.flock_model_transformer import FLockModelTransformer
    from transformer.pfl_transformer import PFLState, VALID_PFL_METHODS

    cfg = OmegaConf.load(a.config)
    seed_everything(int(getattr(cfg.common_args, "random_seed", 42)))

    pfl_method = (str(cfg.train_args.get("pfl_method") or "")).strip().lower()
    is_pfl = pfl_method in VALID_PFL_METHODS

    source_dir = os.path.abspath(a.source)
    source_name = os.path.basename(source_dir.rstrip("/"))
    client_id = a.client_id or f"{source_name}@{socket.gethostname()}#{os.getpid()}"
    cfg_dir = os.path.dirname(os.path.abspath(a.config))
    out_root = a.out_dir or os.path.join(cfg_dir, "output", "clients", source_name)
    os.makedirs(out_root, exist_ok=True)

    base = a.server.rstrip("/")

    print(f"[{client_id}] building CGMTransformer model for cohort '{source_name}'...", flush=True)
    model = FLockModelTransformer(build_client_args(cfg, source_dir, out_root), verbose=False)
    n_train = int(getattr(model, "n_train", 0))

    pfl_state = None
    if is_pfl:
        pfl_state = PFLState(
            method=pfl_method,
            model_owner=model,
            apfl_alpha_init=float(getattr(cfg.train_args, "apfl_alpha_init", 0.25)),
            apfl_alpha_lr=float(getattr(cfg.train_args, "apfl_alpha_lr", 0.1)),
            ditto_prox_mu=float(getattr(cfg.train_args, "ditto_prox_mu", 0.1)),
        )
        print(f"[{client_id}] PFL enabled: method={pfl_method}", flush=True)

    C.http_post_json(f"{base}/register",
                     {"client_id": client_id, "source_name": source_name, "n_train": n_train})
    print(f"[{client_id}] registered with {base} (cohort={source_name}, n_train={n_train})", flush=True)

    while True:
        task = C.http_get_json(f"{base}/task?client_id={client_id}")
        action = task.get("action")

        if action == "done":
            print(f"[{client_id}] server signalled DONE — exiting.", flush=True)
            break
        if action in (None, "wait"):
            time.sleep(a.poll)
            continue

        rnd = task.get("round")
        blob = C.http_get_bytes(f"{base}/model?tag={task['model_tag']}")

        if action == "train":
            if is_pfl:
                local_blob, count, train_loss = pfl_state.train_round(global_blob=blob)
            else:
                local_blob, count, train_losses, _val = model.train(parameters=blob, comm_round_idx=rnd)
                train_loss = float(train_losses[-1]) if train_losses else float("nan")
            url = (f"{base}/train_result?client_id={client_id}&round={rnd}"
                   f"&n_train={count}&train_loss={train_loss}")
            resp = C.http_post_bytes(url, local_blob)
            if not resp.get("ok", False):
                print(f"[{client_id}] round {rnd}: train submit stale ({resp.get('msg')}) — re-polling", flush=True)
            else:
                extra = f" alpha={pfl_state.alpha:.4f}" if is_pfl and pfl_method != "ditto" else ""
                print(f"[{client_id}] round {rnd}: trained (train_loss={train_loss:.5f}, n={count}){extra}", flush=True)

        elif action == "eval":
            if is_pfl:
                # DITTO: run the v proximal update against the freshly aggregated global.
                if pfl_method == "ditto":
                    pfl_state.run_ditto_v_pass(blob)
                deploy_blob = pfl_state.build_deploy_blob(new_global_blob=blob)
                m = model.evaluate_local_val(parameters=deploy_blob)
            else:
                m = model.evaluate_local_val(parameters=blob)
            val_mse = float(m.get("mse_norm", float("inf")))
            val_rmse = float(m.get("rmse_mgdl_30min", float("inf")))
            C.http_post_json(f"{base}/eval_result?client_id={client_id}&round={rnd}",
                             {"val_mse": val_mse, "val_rmse": val_rmse})
            print(f"[{client_id}] round {rnd}: eval val_mse={val_mse:.6f} val_rmse={val_rmse:.3f}", flush=True)
            # PFL clients wait for /round_info to learn whether this round was
            # the new global best; if so, snapshot the deploy blob for use at
            # test. This wait is deliberately UNBOUNDED, exactly like the /task
            # "wait" loop: the round cannot complete without every client, so a
            # timeout here could only ever mean "the federation is stuck", and
            # silently continuing would let this client reach the test phase
            # having never snapshotted its best-round personal model — i.e. it
            # would report test metrics for the wrong model.
            if is_pfl:
                while True:
                    info = C.http_get_json(f"{base}/round_info?round={rnd}")
                    if info.get("complete"):
                        pfl_state.snapshot_if_best(rnd, bool(info.get("is_best")))
                        if info.get("is_best"):
                            print(f"[{client_id}] round {rnd}: new global best — snapshotted deploy blob", flush=True)
                        break
                    time.sleep(a.poll)

        elif action == "test":
            if is_pfl:
                deploy = pfl_state.get_best_deploy_blob() or blob
                m = model.evaluate_local_test(parameters=deploy)
            else:
                m = model.evaluate_local_test(parameters=blob)
            C.http_post_json(f"{base}/test_result?client_id={client_id}",
                             {"metrics": C.json_safe(m)})
            print(f"[{client_id}] TEST: rmse_30={m.get('rmse_mgdl_30min'):.3f} "
                  f"tir_actual={m.get('tir_actual'):.1f}% tir_pred={m.get('tir_predicted'):.1f}%",
                  flush=True)
        else:
            time.sleep(a.poll)


if __name__ == "__main__":
    main()
