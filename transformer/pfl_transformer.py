"""
pfl_transformer.py — Personalised-FL round primitives for the networked client.

This module implements APFL (faithful Local-Descent, Deng-Kamani-Mahdavi 2020,
Alg. 1), an experimental APFL-decoupled variant, and DITTO (Li et al. 2021)
for the single-client-per-process networked setting used by fl_client.py.

Design contract
---------------
- Each FL client process holds ONE PFLState (persistent across rounds).
- The w-path is standard FedAvg on the model's own parameters and is aggregated
  by the server unchanged — this module only adds the personal branches (v_i,
  alpha_i, personal Adam) and the deploy-blob machinery.
- The client owns the deploy blob (mixture vbar for apfl/decoupled, personal v
  for ditto) and its "best" snapshot; the server's role for PFL is unchanged
  aside from a new /round_info?round=r endpoint used by the client to learn
  whether the just-finished round was the new global best.

All three algorithms use the model owner's:
  - .model, .device, .quantile_loss, .train_loader
  - ._build_optimizer(), ._load_global_params(blob, context)
  - .max_steps_per_epoch, .num_epochs

and return exactly the same shape (w_blob, n_train, last_train_loss) that the
FedAvg train() path returns, so the server contract is unchanged.
"""

from __future__ import annotations

import copy
import io
from typing import Dict, Optional

import torch
import torch.nn as nn


VALID_PFL_METHODS = {"apfl", "apfl_decoupled", "ditto"}


def _serialise(state_dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def _deserialise(blob: bytes):
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


def build_mixture_blob(v_state_cpu: Dict[str, torch.Tensor],
                       global_blob: bytes, alpha: float) -> bytes:
    """vbar = alpha * v + (1 - alpha) * global, for float params.
    Non-float keys (buffers, ints) are taken from the global. Used for APFL
    and APFL-decoupled deploy/eval."""
    g = _deserialise(global_blob)
    out = {}
    for k, gv in g.items():
        if (k in v_state_cpu
                and gv.is_floating_point()
                and v_state_cpu[k].shape == gv.shape):
            out[k] = alpha * v_state_cpu[k] + (1.0 - alpha) * gv
        else:
            out[k] = gv
    return _serialise(out)


class PFLState:
    """Persistent per-client PFL state. One instance per client process."""

    def __init__(self, method: str, model_owner,
                 apfl_alpha_init: float = 0.25,
                 apfl_alpha_lr: float = 0.1,
                 ditto_prox_mu: float = 0.1):
        method = str(method).lower().strip()
        if method not in VALID_PFL_METHODS:
            raise ValueError(f"Unknown pfl_method '{method}'. "
                             f"Expected one of {sorted(VALID_PFL_METHODS)}.")
        self.method = method
        self._owner = model_owner   # weak coupling; we access owner.model etc.

        # APFL / APFL-decoupled
        self.alpha = float(apfl_alpha_init)
        self.alpha_lr = float(apfl_alpha_lr)

        # Scratch model that we overwrite each step to carry the mixture vbar
        # for the alpha-correlation forward. Sized identically to owner.model.
        self._vmod = None       # built lazily on first train

        # Persistent personal state
        # - APFL: v held as a dict of CPU tensors (updated in the training loop)
        # - APFL-decoupled: v held as a persistent nn.Module (with its own Adam)
        # - DITTO: v held as a persistent nn.Module (with its own Adam)
        self._v_state_cpu: Optional[Dict[str, torch.Tensor]] = None  # apfl
        self._pmod: Optional[nn.Module] = None                        # decoupled / ditto
        self._popt: Optional[torch.optim.Optimizer] = None            # decoupled / ditto
        self.ditto_prox_mu = float(ditto_prox_mu)

        # The deploy blob used for eval / test. Snapshotted whenever the server
        # tells us the just-finished round was the new global best.
        self.last_deploy_blob: Optional[bytes] = None
        self.best_deploy_blob: Optional[bytes] = None
        self.best_round: Optional[int] = None

    # ── init helpers ────────────────────────────────────────────────────────
    def _ensure_scratch_vmod(self):
        if self._vmod is None:
            self._vmod = copy.deepcopy(self._owner.model).to(self._owner.device)

    def _ensure_personal_module(self, lr: float, wd: float):
        """Build the persistent personal model + Adam once (decoupled + ditto)."""
        if self._pmod is None:
            self._pmod = copy.deepcopy(self._owner.model).to(self._owner.device)
            self._popt = torch.optim.Adam(self._pmod.parameters(),
                                          lr=lr, weight_decay=wd)

    def _batch_iter(self):
        """Yield up to max_steps_per_epoch batches for num_epochs, matching
        the FedAvg client's inner loop budget."""
        cap = self._owner.max_steps_per_epoch or 10 ** 9
        for _ in range(int(self._owner.num_epochs)):
            for step, batch in enumerate(self._owner.train_loader, 1):
                if step > cap:
                    break
                yield batch

    def _to_device(self, batch):
        return [t.to(self._owner.device) for t in batch]

    # ── round entry point ──────────────────────────────────────────────────
    def train_round(self, global_blob: Optional[bytes]) -> tuple:
        """Runs one PFL local round and returns (w_blob, n_train, last_loss).
        The returned w_blob is what the server aggregates via FedAvg — the
        personal branch stays on the client."""
        if self.method == "apfl":
            return self._train_apfl(global_blob)
        if self.method == "apfl_decoupled":
            return self._train_apfl_decoupled(global_blob)
        if self.method == "ditto":
            # DITTO's v-pass is deferred until the client sees the freshly
            # aggregated global (in eval phase). Here we only run the standard
            # FedAvg w-training.
            return self._train_ditto_w(global_blob)
        raise RuntimeError(f"unreachable: unknown pfl method {self.method!r}")

    # ── APFL faithful Local-Descent ────────────────────────────────────────
    def _train_apfl(self, global_blob: Optional[bytes]) -> tuple:
        owner = self._owner
        M = owner.model
        device = owner.device
        M.to(device)
        if global_blob is not None:
            owner._load_global_params(global_blob, context="apfl")

        self._ensure_scratch_vmod()
        vmod = self._vmod
        vmod.load_state_dict(M.state_dict())    # sync buffers

        name2p = dict(M.named_parameters())
        vmod_params = dict(vmod.named_parameters())

        # v_i on first round starts equal to the shared init (= current w).
        if self._v_state_cpu is None:
            v_dev = {n: p.detach().clone() for n, p in M.named_parameters()}
        else:
            v_dev = {n: t.to(device) for n, t in self._v_state_cpu.items()}

        opt_w = owner._build_optimizer()
        lr = float(owner.train_args.proposer_learning_rate)  # for v update
        alpha = float(self.alpha)
        M.train(); vmod.train()

        last_loss = float("nan")
        alpha_num, alpha_cnt = 0.0, 0

        for batch in self._batch_iter():
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, _cm, _cs, _ = self._to_device(batch)

            # (6) gradient of the GLOBAL copy at w^{t-1}
            opt_w.zero_grad()
            loss_w = owner.quantile_loss(M(x_enc, x_mark_enc, x_dec, x_mark_dec), y)
            loss_w.backward()
            last_loss = float(loss_w.item())

            # build vbar^{t-1} = alpha*v^{t-1} + (1-alpha)*w^{t-1} into vmod
            with torch.no_grad():
                for n, p in vmod_params.items():
                    p.copy_(alpha * v_dev[n] + (1.0 - alpha) * name2p[n].detach())
            vmod.zero_grad()
            loss_v = owner.quantile_loss(vmod(x_enc, x_mark_enc, x_dec, x_mark_dec), y)
            loss_v.backward()

            # (10) alpha correlation <v - w, grad f(vbar)> using pre-step values
            with torch.no_grad():
                dot = 0.0
                for n, p in vmod_params.items():
                    if p.grad is not None:
                        dot += float(torch.sum((v_dev[n] - name2p[n].detach()) * p.grad))
            alpha_num += dot; alpha_cnt += 1

            # (6) step w; (7) step v ← v - eta*alpha*grad f(vbar)
            nn.utils.clip_grad_norm_(M.parameters(), 1.0)
            opt_w.step()
            with torch.no_grad():
                for n, p in vmod_params.items():
                    if p.grad is not None:
                        v_dev[n].add_(p.grad, alpha=-(lr * alpha))

        if alpha_cnt > 0:  # eq (10), once per round
            alpha = float(min(1.0, max(0.0, alpha - self.alpha_lr * (alpha_num / alpha_cnt))))

        self.alpha = alpha
        self._v_state_cpu = {n: t.detach().cpu() for n, t in v_dev.items()}
        w_blob = _serialise(M.state_dict())
        return w_blob, owner.n_train, last_loss

    # ── APFL-DECOUPLED: v trains on own loss with own Adam ─────────────────
    def _train_apfl_decoupled(self, global_blob: Optional[bytes]) -> tuple:
        owner = self._owner
        M = owner.model; device = owner.device
        M.to(device)
        if global_blob is not None:
            owner._load_global_params(global_blob, context="apfl_decoupled")

        lr = float(owner.train_args.proposer_learning_rate)
        wd = float(getattr(owner.train_args, "proposer_train_weight_decay", 0.0))
        self._ensure_personal_module(lr=lr, wd=wd)
        self._ensure_scratch_vmod()
        pmod = self._pmod; popt = self._popt; vmod = self._vmod

        opt_w = owner._build_optimizer()
        name2p  = dict(M.named_parameters())
        pmod_p  = dict(pmod.named_parameters())
        vmod_p  = dict(vmod.named_parameters())

        M.train(); pmod.train(); vmod.train()
        alpha = float(self.alpha)
        alpha_num, alpha_cnt = 0.0, 0
        last_loss = float("nan")

        for batch in self._batch_iter():
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, _cm, _cs, _ = self._to_device(batch)

            # (alpha) correlation at the pre-step mixture
            with torch.no_grad():
                for n, p in vmod_p.items():
                    p.copy_(alpha * pmod_p[n].detach() + (1.0 - alpha) * name2p[n].detach())
            vmod.zero_grad()
            owner.quantile_loss(vmod(x_enc, x_mark_enc, x_dec, x_mark_dec), y).backward()
            with torch.no_grad():
                dot = 0.0
                for n, p in vmod_p.items():
                    if p.grad is not None:
                        dot += float(torch.sum((pmod_p[n] - name2p[n]) * p.grad))
            alpha_num += dot; alpha_cnt += 1

            # (w) global path — identical to fedavg baseline
            opt_w.zero_grad()
            lw = owner.quantile_loss(M(x_enc, x_mark_enc, x_dec, x_mark_dec), y)
            lw.backward()
            last_loss = float(lw.item())
            nn.utils.clip_grad_norm_(M.parameters(), 1.0)
            opt_w.step()

            # (v) personal path — own loss, own persistent Adam, full lr
            popt.zero_grad()
            lv = owner.quantile_loss(pmod(x_enc, x_mark_enc, x_dec, x_mark_dec), y)
            lv.backward()
            nn.utils.clip_grad_norm_(pmod.parameters(), 1.0)
            popt.step()

        if alpha_cnt > 0:
            alpha = float(min(1.0, max(0.0, alpha - self.alpha_lr * (alpha_num / alpha_cnt))))
        self.alpha = alpha
        w_blob = _serialise(M.state_dict())
        return w_blob, owner.n_train, last_loss

    # ── DITTO: standard FedAvg on w, then a separate prox pass on v ────────
    def _train_ditto_w(self, global_blob: Optional[bytes]) -> tuple:
        """DITTO's w-path is exactly the FedAvg baseline: load global, run the
        standard client train, return the resulting w blob. The v-pass happens
        AFTER aggregation in run_ditto_v_pass()."""
        owner = self._owner
        # v_i^(0) must be the SHARED init w^(0), matching the reference
        # implementation which deep-copies every client's personal model from
        # the untrained model before the round loop begins. We must therefore
        # materialise _pmod here, BEFORE owner.train() mutates owner.model in
        # place — otherwise v_i^(0) would silently become "w^(0) + one round of
        # purely local training", which is a different algorithm.
        if self._pmod is None:
            if global_blob is not None:
                owner._load_global_params(global_blob, context="ditto_v_init")
            self._ensure_personal_module(
                lr=float(owner.train_args.proposer_learning_rate),
                wd=float(getattr(owner.train_args, "proposer_train_weight_decay", 0.0)),
            )
        # Reuse the model owner's own vanilla train (which handles all the
        # existing bells and whistles: MLDG off, FedProx off, clipping, etc.)
        w_blob, n_train, train_losses, _ = owner.train(
            parameters=global_blob, comm_round_idx=None)
        last_loss = train_losses[-1] if train_losses else float("nan")
        return w_blob, n_train, last_loss

    def run_ditto_v_pass(self, new_global_blob: bytes):
        """Run the personal-branch (v) proximal update AFTER the server has
        aggregated the new global. Call this from the client's eval phase
        before evaluating v."""
        owner = self._owner
        device = owner.device
        lr = float(owner.train_args.proposer_learning_rate)
        wd = float(getattr(owner.train_args, "proposer_train_weight_decay", 0.0))
        mu = float(self.ditto_prox_mu)
        self._ensure_personal_module(lr=lr, wd=wd)
        pmod = self._pmod; popt = self._popt

        gsd = _deserialise(new_global_blob)
        w_anchor = {nm: gsd[nm].to(device)
                    for nm, _ in pmod.named_parameters() if nm in gsd}
        pmod.train()

        for batch in self._batch_iter():
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, _cm, _cs, _ = self._to_device(batch)
            popt.zero_grad()
            lv = owner.quantile_loss(pmod(x_enc, x_mark_enc, x_dec, x_mark_dec), y)
            if mu > 0:
                prox = 0.0
                for nm, p in pmod.named_parameters():
                    if nm in w_anchor:
                        prox = prox + ((p - w_anchor[nm]) ** 2).sum()
                lv = lv + 0.5 * mu * prox
            lv.backward()
            nn.utils.clip_grad_norm_(pmod.parameters(), 1.0)
            popt.step()

    # ── eval-time deploy-blob construction ─────────────────────────────────
    def build_deploy_blob(self, new_global_blob: bytes) -> bytes:
        """Build the blob to evaluate at this round's eval phase.

        - apfl:          vbar = alpha*v + (1-alpha) * new_global
        - apfl_decoupled: same as apfl (v is the persistent pmod)
        - ditto:         personal v directly (no mixture)
        Sets self.last_deploy_blob and returns it."""
        if self.method == "apfl":
            if self._v_state_cpu is None:
                # first-round edge case: v == global, so vbar == global
                self.last_deploy_blob = new_global_blob
            else:
                self.last_deploy_blob = build_mixture_blob(
                    self._v_state_cpu, new_global_blob, self.alpha)
        elif self.method == "apfl_decoupled":
            if self._pmod is None:
                self.last_deploy_blob = new_global_blob
            else:
                v_cpu = {nm: p.detach().cpu() for nm, p in self._pmod.named_parameters()}
                self.last_deploy_blob = build_mixture_blob(
                    v_cpu, new_global_blob, self.alpha)
        elif self.method == "ditto":
            if self._pmod is None:
                self.last_deploy_blob = new_global_blob
            else:
                self.last_deploy_blob = _serialise(self._pmod.state_dict())
        else:
            raise RuntimeError(f"unreachable: {self.method!r}")
        return self.last_deploy_blob

    # ── best-round snapshotting (called on is_best from /round_info) ───────
    def snapshot_if_best(self, rnd: int, is_best: bool):
        if is_best and self.last_deploy_blob is not None:
            self.best_deploy_blob = self.last_deploy_blob
            self.best_round = int(rnd)

    def get_best_deploy_blob(self) -> Optional[bytes]:
        """Returned at test phase in place of the server's best global model."""
        return self.best_deploy_blob or self.last_deploy_blob
