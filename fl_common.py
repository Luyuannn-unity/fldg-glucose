"""
Shared helpers for the from-scratch CGMTransformer federated-learning system.

The CGMTransformer model + data loaders + training loop are VENDORED into this
folder (the `transformer/` package) — this project is fully self-contained and has
no dependency on any external repo. On top of that we add the bits a real
networked FL needs: minting initial weights without a dataset, plain FedAvg
over serialized state_dicts, and tiny stdlib HTTP helpers (no requests, no
Flower — "from scratch").
"""

import io
import os
import sys
import json
import time
import urllib.request
import urllib.error

# Make the vendored `transformer` and `flock_sdk` packages importable no matter
# what the current working directory is (this file lives next to them).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# ───────────────────────────── model / weights ──────────────────────────────
# These run server-side and need NO dataset and NO GPU — just the model graph,
# so the server can produce byte-identical initial weights and broadcast them.

def build_model_cfg(cfg):
    """Replicate FLockModelTransformer._make_model_cfg from an OmegaConf cfg.

    Kept in lock-step with that method so server-built init weights match the
    architecture the clients build.
    """
    ma = cfg.model_args
    da = cfg.data_args
    loss_type = str(getattr(cfg.train_args, "proposer_loss_type", "quantile")).lower()

    class _Cfg:
        pass

    c = _Cfg()
    c.model_name = getattr(ma, "model_name", "transformer")
    c.enc_in = getattr(ma, "enc_in", 1)
    c.dec_in = getattr(ma, "dec_in", 1)
    c.c_out = 1 if loss_type == "mse" else getattr(ma, "c_out", 5)
    c.d_model = getattr(ma, "d_model", 128)
    c.n_heads = getattr(ma, "n_heads", 8)
    c.e_layers = getattr(ma, "e_layers", 3)
    c.d_layers = getattr(ma, "d_layers", 1)
    c.d_ff = getattr(ma, "d_ff", 2048)
    c.dropout = getattr(ma, "dropout", 0.05)
    c.embed = getattr(ma, "embed", "cycF")
    c.freq = getattr(ma, "freq", "cyc")
    c.activation = getattr(ma, "activation", "relu")
    c.factor = getattr(ma, "factor", 5)
    c.full_attention = getattr(ma, "full_attention", True)
    c.output_attention = getattr(ma, "output_attention", False)
    c.pred_len = getattr(da, "pred_len", 24)
    c.seq_len = getattr(da, "seq_len", 72)
    c.patch_size = getattr(ma, "patch_size", 6)
    return c


def build_initial_state_bytes(cfg) -> bytes:
    """Seed RNG from config, build the model graph, return its serialized
    state_dict. This is the round-1 global model every client receives, so all
    clients start from identical weights regardless of their hardware."""
    import random
    import numpy as np
    import torch

    from transformer.model_hub_transformer import build_transformer

    seed = int(getattr(cfg.common_args, "random_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = build_transformer(build_model_cfg(cfg))
    return serialize_state_dict(model.state_dict())


def serialize_state_dict(sd) -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save(sd, buf)
    return buf.getvalue()


def deserialize_state_dict(blob: bytes):
    import torch

    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


def fedavg(blobs, counts) -> bytes:
    """Sample-weighted FedAvg over serialized state_dicts.

    Identical math to FLockModelTransformer.aggregate(): weight each client by its
    training-sample count. FedProx and MLDG also aggregate this way (they only
    change client-side training), so the server uses this for all three methods.
    """
    if not blobs:
        raise ValueError("fedavg: no client updates to aggregate")
    if len(blobs) == 1:
        return blobs[0]

    total = float(sum(counts))
    if total <= 0:
        weights = [1.0 / len(blobs)] * len(blobs)
    else:
        weights = [c / total for c in counts]

    sds = [deserialize_state_dict(b) for b in blobs]
    agg = {}
    for key in sds[0]:
        agg[key] = sum(sd[key] * w for sd, w in zip(sds, weights))
    return serialize_state_dict(agg)


# ───────────────────────────── HTTP (stdlib) ────────────────────────────────
# Minimal urllib wrappers with retry, so clients can be started before the
# server is up and survive transient network blips.

def _request(url, data=None, headers=None, method=None,
             timeout=600, retries=60, backoff=3.0):
    headers = headers or {}
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            # 503 = server busy / not ready: retry. Other 4xx/5xx are real.
            body = b""
            try:
                body = e.read()
            except Exception:
                pass
            if e.code == 503 and attempt < retries - 1:
                last_err = e
                time.sleep(backoff)
                continue
            raise RuntimeError(f"HTTP {e.code} for {url}: {body.decode(errors='replace')}") from e
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            # connection refused / reset / timeout — server may be starting up
            last_err = e
            time.sleep(backoff)
    raise RuntimeError(f"request to {url} failed after {retries} attempts: {last_err}")


def http_get_json(url, **kw):
    _, body = _request(url, method="GET", **kw)
    return json.loads(body.decode())


def http_get_bytes(url, **kw):
    _, body = _request(url, method="GET", **kw)
    return body


def http_post_json(url, obj, **kw):
    data = json.dumps(obj).encode()
    _, body = _request(url, data=data,
                       headers={"Content-Type": "application/json"},
                       method="POST", **kw)
    return json.loads(body.decode()) if body else {}


def http_post_bytes(url, data, **kw):
    _, body = _request(url, data=data,
                       headers={"Content-Type": "application/octet-stream"},
                       method="POST", **kw)
    return json.loads(body.decode()) if body else {}


def json_safe(d: dict) -> dict:
    """Coerce a metrics dict to JSON-serialisable floats/strings."""
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v if isinstance(v, (str, int, bool, type(None))) else str(v)
    return out
