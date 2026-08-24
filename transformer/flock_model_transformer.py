"""
FLock federated-learning wrapper for CGMTransformer blood-glucose prediction.

Implements the three-method FlockModel interface:
  train(parameters)      -> bytes          (FedAvg proposer side)
  evaluate(parameters)   -> float          (voter RMSE at 30-min horizon)
  aggregate(params_list) -> bytes          (FedAvg server side)

Data contract
-------------
Each FL client is initialised with one parquet file that contains at least:
  - CGM   : continuous glucose monitor reading (mg/dL)
  - date  : timestamp string

The data utilities handle windowing and z-score normalisation per window.
Model output is 5-quantile predictions; RMSE is computed on the median (q=0.5).
"""

import sys
import io
import os
import csv
import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# matplotlib is only needed by the (optional) plotting methods, which the
# real-FL client/server never call. Import it lazily so a missing/broken
# matplotlib install doesn't block training/eval.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None
from loguru import logger
from typing import List, Optional, Tuple

from flock_sdk import FlockModel
from .data_utils_transformer import (
    CGMTransformerDataConfig,
    compute_stats_from_metabonet_split,
    create_lazy_loaders_from_metabonet,
)
from .model_hub_transformer import build_transformer


# ---------------------------------------------------------------------------
# Quantile loss (identical to finetune_metabonet.py)
# ---------------------------------------------------------------------------

class QuantileLoss:
    """Pinball / quantile loss over multiple quantiles."""

    def __init__(self, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
        self.quantiles = list(quantiles)

    def __call__(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred : (B, pred_len, n_quantiles)
            target : (B, pred_len, 1)
        """
        mask = target[..., 1:2] if target.shape[-1] > 1 else None
        target = target[..., :1]
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., [i]]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        per_elem = torch.sum(torch.cat(losses, dim=1), dim=1)
        if mask is None:
            return torch.mean(per_elem)
        return (per_elem * mask).sum() / mask.sum().clamp(min=1.0)

    def median(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Return the median (q=0.5) quantile slice."""
        idx = self.quantiles.index(0.5)
        return y_pred[..., [idx]]


class MSELossAdapter:
    """MSE loss with the same interface as QuantileLoss (drop-in replacement).

    Model is expected to output (B, pred_len, 1) when this adapter is used
    (i.e. c_out=1).  `median()` returns the single output unchanged so the
    RMSE-eval call sites keep working without modification.
    """

    def __init__(self):
        self.quantiles = [0.5]

    def __call__(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # target may carry the cgm_real mask as channel 1: loss is then the
        # masked mean over real (non-interpolated) samples only.
        if target.shape[-1] > 1:
            mask = target[..., 1:2]
            se = (y_pred - target[..., :1]) ** 2
            return (se * mask).sum() / mask.sum().clamp(min=1.0)
        return torch.mean((y_pred - target) ** 2)

    def median(self, y_pred: torch.Tensor) -> torch.Tensor:
        return y_pred[..., [0]]


# ---------------------------------------------------------------------------
# Main FL model class
# ---------------------------------------------------------------------------

class FLockModelTransformer(FlockModel):
    """
    Federated-learning wrapper around the CGMTransformer transformer model.

    One instance = one FL client (one patient's parquet data).
    """

    def __init__(self, args, verbose: bool = False):
        super().__init__()
        self.args = args
        self._configure_logger(verbose)
        self._configure_parameters(args)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"CGMTransformer FL client — device: {self.device}")

        # Build model config from yaml args
        self.model_cfg = self._make_model_cfg(args)

        # Build model
        self.model = build_transformer(self.model_cfg)
        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"\n{'='*60}\n"
            f"{self.model}\n"
            f"{'='*60}\n"
            f"Total parameters:     {total_params:>12,}\n"
            f"Trainable parameters: {trainable_params:>12,}\n"
            f"{'='*60}"
        )

        # Build data config (no train_ratio/val_ratio — splits come from seg_split_map.json)
        data_cfg = self._data_cfg = CGMTransformerDataConfig(
            seq_len=getattr(args.data_args, 'seq_len', 256),
            label_len=getattr(args.data_args, 'label_len', 6),
            pred_len=getattr(args.data_args, 'pred_len', 24),
            stride=getattr(args.data_args, 'stride', 1),
            max_gap_steps=getattr(args.data_args, 'max_gap_steps', 6),
            sampling_minutes=getattr(args.data_args, 'sampling_minutes', 5),
            use_bolus=getattr(args.data_args, 'use_bolus', False),
        )

        if data_cfg.use_bolus and getattr(args.model_args, 'enc_in', 1) != 2:
            raise ValueError(
                f"data_args.use_bolus is True but model_args.enc_in is "
                f"{getattr(args.model_args, 'enc_in', 1)}; set enc_in: 2."
            )

        # Normalisation stats
        if self.global_cgm_mean is not None and self.global_cgm_std is not None:
            g_mean = float(self.global_cgm_mean)
            g_std  = float(self.global_cgm_std)
        else:
            g_mean, g_std = compute_stats_from_metabonet_split(self.source_dir, split='train')

        # Lazy DataLoaders backed by mmap'd packed segments + window_index files
        # (mirrors LoRA's MetaboNetDataset / LoRAWindowDataset approach).
        self.train_loader, self.val_loader, self.test_loader = (
            create_lazy_loaders_from_metabonet(
                self.source_dir, data_cfg,
                cgm_mean=g_mean, cgm_std=g_std,
                train_batch_size=self.train_batch_size,
                val_batch_size=self.val_batch_size,
                test_batch_size=self.test_batch_size,
            )
        )
        if self.train_loader is None:
            raise RuntimeError(f"No training windows for {self.source_dir}")
        self.n_train = len(self.train_loader.dataset)

        # MLDG builds its support/query split from patient identity, which lives
        # in column 2 of window_index_<split>.npy. With a 2-column index every
        # window reports pat_idx=-1, so the batch looks like a single domain and
        # each MLDG step silently degrades to a vanilla step — an MLDG run that
        # is really FedAvg. Fail loudly instead.
        if self.mldg and not getattr(self.train_loader.dataset, "has_patient_column", False):
            raise RuntimeError(
                f"train_args.mldg is enabled but {self.source_dir}'s "
                f"window_index_train.npy has no patient column. MLDG needs an "
                f"int32 [N, 3] index of (seg_idx, start_pos, patient_int_idx); "
                f"this index is [N, 2]. Rebuild it with data_pipeline/build_metabonet.py "
                f"(see data_pipeline/DATASET.md section 6), or set mldg: false."
            )

        src_name = os.path.basename(self.source_dir)
        logger.info(
            f"{src_name}: train={self.n_train} windows  "
            f"mean={g_mean:.1f}  std={g_std:.1f}"
        )

        loss_type = str(getattr(args.train_args, 'proposer_loss_type', 'quantile')).lower()
        if loss_type == 'mse':
            self.quantile_loss = MSELossAdapter()
            logger.info("Loss: MSE (single-head, c_out=1)")
        else:
            self.quantile_loss = QuantileLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        # Index of the 30-min prediction step (6 steps at 5-min resolution = 30 min)
        self.horizon_30min_idx = getattr(args.evaluation_args, 'horizon_30min_idx', 5)

        # Populated at the end of each MLDG-enabled train() round.
        self.last_mldg_stats: Optional[dict] = None

    # ------------------------------------------------------------------ init

    def _configure_logger(self, verbose: bool):
        log_level = "DEBUG" if verbose else "INFO"
        logger.remove()
        logger.add(sys.stdout, level=log_level)

    def _configure_parameters(self, args):
        self.source_dir  = str(args.data_args.source_dir)
        self.source_name = os.path.basename(self.source_dir.rstrip("/"))
        self.normalization_mode = str(
            getattr(args.data_args, 'normalization_mode', 'per_client')
        ).lower()
        self.global_cgm_mean = getattr(args.data_args, 'global_cgm_mean', None)
        self.global_cgm_std  = getattr(args.data_args, 'global_cgm_std',  None)

        ta = args.train_args
        self.train_args = ta   # keep the full block so PFL / future extensions can read knobs directly
        self.train_batch_size  = ta.proposer_train_batch_size
        self.val_batch_size    = ta.proposer_val_batch_size
        self.test_batch_size   = ta.proposer_test_batch_size
        self.num_epochs          = ta.proposer_num_epochs
        self.max_steps_per_epoch = getattr(ta, 'proposer_max_steps_per_epoch', None)
        self.learning_rate       = ta.proposer_learning_rate
        self.weight_decay      = ta.proposer_train_weight_decay
        self.optimizer_name    = ta.proposer_train_optimizer_name
        self.scheduler_type    = ta.proposer_train_lr_scheduler_type
        self.save_steps        = ta.proposer_save_steps
        self.early_stopping    = getattr(ta, 'early_stopping', False)
        self.total_rounds      = int(getattr(ta, 'num_communication_rounds', 50))

        # MLDG
        self.mldg             = getattr(ta, 'mldg', False)
        self.mldg_first_order = getattr(ta, 'mldg_first_order', True)  # default True: no higher needed
        self.mldg_n_support   = getattr(ta, 'mldg_n_support', 2)
        self.mldg_n_query     = getattr(ta, 'mldg_n_query', 1)
        self.mldg_inner_iters = getattr(ta, 'mldg_inner_iters', 1)
        self.mldg_trade_off   = getattr(ta, 'mldg_trade_off', 1.0)
        self.mldg_inner_lr    = getattr(ta, 'mldg_inner_lr', None)  # None = use outer LR
        # Reference-recipe (CGMTransformer-paper) second-order MLDG: the differentiable
        # inner loop reuses the OUTER optimizer (so SGD+momentum+nesterov+wd at
        # the outer LR) instead of a fresh plain SGD at mldg_inner_lr.  Only
        # consulted by _train_step_mldg_ref; the fo/so paths are unaffected.
        self.mldg_inner_reuse_outer_opt = bool(
            getattr(ta, 'mldg_inner_reuse_outer_opt', False))
        # H1 verification: per-MLDG-step grad alignment (cos, norms) logging.
        self.mldg_log_grad_alignment = bool(getattr(ta, 'mldg_log_grad_alignment', False))

        # Partial-FL: which state_dict keys are aggregated/broadcast.  Empty
        # list (default) means "federate everything" — backward compatible.
        # Match is by prefix: any(key.startswith(p) for p in prefixes).
        # Common recipe: ["enc_embedding.", "encoder."]  →  encoder federated,
        # decoder client-specific.
        prefixes = getattr(ta, 'federated_param_prefixes', None) or []
        self.federated_param_prefixes = [str(p) for p in prefixes]

        # Federated algorithm: fedavg | fedprox | scaffold
        self.fed_algo = str(getattr(ta, 'federated_optimizer_name', 'fedavg')).lower()
        if self.fed_algo not in ('fedavg', 'fedprox', 'scaffold'):
            logger.warning(
                f"Unknown federated_optimizer_name '{self.fed_algo}', using 'fedavg'"
            )
            self.fed_algo = 'fedavg'
        self.fedprox_mu    = float(getattr(ta, 'fedprox_mu', 0.01))
        self.scaffold_lr_g = float(getattr(ta, 'scaffold_lr_g', 1.0))
        if self.mldg and self.fed_algo != 'fedavg':
            raise ValueError(
                f"federated_optimizer_name='{self.fed_algo}' is incompatible with "
                f"mldg=True; the FedProx/SCAFFOLD code paths only apply to the "
                f"vanilla SGD step. Set mldg=False or use fedavg."
            )
        # Persistent SCAFFOLD state (kept on CPU between rounds).
        self._scaffold_c_local:  Optional[dict] = None
        self._scaffold_c_global: Optional[dict] = None
        # Stash for the orchestrator to harvest after train().
        self.last_scaffold_delta_c: Optional[dict] = None

        ea = args.evaluation_args
        self.voter_val_set_size = ea.voter_val_set_size

        tr = args.tracking_args
        self.plot_root = str(getattr(tr, 'out_put_root', 'output/transformer_v1'))

        if self.normalization_mode == 'global':
            if self.global_cgm_mean is not None and self.global_cgm_std is not None:
                self.global_cgm_mean = float(self.global_cgm_mean)
                self.global_cgm_std  = max(float(self.global_cgm_std), 1e-6)
            logger.info(
                f"Using global normalization stats mean={self.global_cgm_mean:.2f}, "
                f"std={self.global_cgm_std:.2f}"
            )
        else:
            self.global_cgm_mean = None
            self.global_cgm_std = None
            logger.info("Using per-client normalization stats")

    def _make_model_cfg(self, args):
        """Build a simple namespace object from yaml model_args."""
        ma = args.model_args
        da = args.data_args

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.model_name = getattr(ma, 'model_name', 'transformer')
        cfg.enc_in     = getattr(ma, 'enc_in', 1)
        cfg.dec_in     = getattr(ma, 'dec_in', 1)
        # MSE loss path uses a single-output head, regardless of yaml c_out.
        _loss_type = str(getattr(args.train_args, 'proposer_loss_type', 'quantile')).lower()
        cfg.c_out  = 1 if _loss_type == 'mse' else getattr(ma, 'c_out', 5)
        cfg.d_model    = getattr(ma, 'd_model', 128)
        cfg.n_heads    = getattr(ma, 'n_heads', 8)
        cfg.e_layers   = getattr(ma, 'e_layers', 3)
        cfg.d_layers   = getattr(ma, 'd_layers', 1)
        cfg.d_ff       = getattr(ma, 'd_ff', 2048)
        cfg.dropout    = getattr(ma, 'dropout', 0.05)
        cfg.embed      = getattr(ma, 'embed', 'cycF')
        cfg.freq       = getattr(ma, 'freq', 'cyc')
        cfg.activation = getattr(ma, 'activation', 'relu')
        cfg.factor     = getattr(ma, 'factor', 5)
        cfg.full_attention     = getattr(ma, 'full_attention', True)
        cfg.output_attention   = getattr(ma, 'output_attention', False)
        cfg.pred_len   = getattr(da, 'pred_len', 24)
        # Used only by transformer_patch; harmless for the per-timestep variant.
        cfg.seq_len    = getattr(da, 'seq_len', 72)
        cfg.patch_size = getattr(ma, 'patch_size', 6)
        return cfg

    # ------------------------------------------------------------------ partial-FL helpers

    def _is_federated_key(self, key: str) -> bool:
        """True iff `key` should be averaged across clients."""
        if not self.federated_param_prefixes:
            return True
        return any(key.startswith(p) for p in self.federated_param_prefixes)

    def _filter_state_dict_federated(self, sd: dict) -> dict:
        """Subset of `sd` to keys that are federated (no copy when full FL)."""
        if not self.federated_param_prefixes:
            return sd
        return {k: v for k, v in sd.items() if self._is_federated_key(k)}

    def _federated_named_parameters(self):
        """Iterator over (name, param) for trainable, federated params.

        Used by FedProx / SCAFFOLD so the prox term and control variates
        are restricted to params that actually have a global counterpart.
        """
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if not self._is_federated_key(n):
                continue
            yield n, p

    def _load_global_params(self, blob: Optional[bytes], context: str) -> None:
        """Load `blob` into self.model, filtering to federated keys when partial FL.

        Centralised so all six call sites (train + 4× evaluate + plot)
        stay consistent.  Tolerates missing/extra keys when partial FL is
        on (the blob may legitimately contain only federated keys).
        """
        if blob is None:
            return
        try:
            sd = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        except Exception as e:
            logger.warning(f"{context}: could not deserialize parameters: {e}")
            return
        try:
            sd = self._filter_state_dict_federated(sd)
            strict = not self.federated_param_prefixes
            self.model.load_state_dict(sd, strict=strict)
        except Exception as e:
            logger.warning(f"{context}: could not load parameters: {e}")

    def get_non_federated_state(self) -> dict:
        """Deep-cloned dict of just the non-federated keys (CPU).

        Used by the orchestrator to snapshot or average per-client decoders.
        Returns {} when partial FL is off.
        """
        if not self.federated_param_prefixes:
            return {}
        return {
            k: v.detach().clone().cpu()
            for k, v in self.model.state_dict().items()
            if not self._is_federated_key(k)
        }

    def set_non_federated_state(self, sd: Optional[dict]) -> None:
        """Overwrite this client's non-federated weights.

        Used to give OOD eval-only clients a meaningful decoder
        (typically the average of train-client decoders).
        """
        if not sd:
            return
        try:
            self.model.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.warning(f"set_non_federated_state: load failed: {e}")

    # ------------------------------------------------------------------ init_dataset

    def init_dataset(self, dataset_path: str) -> None:
        """
        Required by FlockModel interface.
        Dataset is already initialised in __init__ from the YAML config,
        so this is a no-op in the CGMTransformer template.
        """
        pass

    # ------------------------------------------------------------------ train

    def train(self, parameters: Optional[bytes] = None,
              comm_round_idx: Optional[int] = None) -> Tuple[bytes, int, list, list]:
        """
        Run one round of local training.

        Args:
            parameters    : serialised global model weights (None = first round)
            comm_round_idx: federation round number (informational)

        Returns:
            (serialised_weights, n_train_samples, train_losses, val_losses)
        """
        if comm_round_idx is None:
            import time; comm_round_idx = int(time.time())

        # Load global parameters if provided.  With partial FL, only federated
        # keys are overwritten; client-specific keys persist from the previous
        # round on this client's self.model.
        if parameters is not None:
            self._load_global_params(parameters, context="train")
            if self.federated_param_prefixes:
                logger.info(
                    f"Loaded global federated params "
                    f"(prefixes={self.federated_param_prefixes})"
                )
            else:
                logger.info("Loaded global model parameters")

        self.model.to(self.device)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, comm_round_idx)

        # ---- FedProx / SCAFFOLD round-start setup -----------------------
        # Snapshot of x = global parameters (on device).  Needed by FedProx's
        # proximal term and by SCAFFOLD's option-II local-c update.
        # NB: FedProx / SCAFFOLD operate ONLY on federated trainable params —
        # client-specific params have no global counterpart, so they don't
        # belong in the prox term or the control variate.
        x_snapshot: Optional[dict] = None
        if self.fed_algo in ('fedprox', 'scaffold'):
            x_snapshot = {
                n: p.detach().clone()
                for n, p in self._federated_named_parameters()
            }
        # SCAFFOLD device-side control variates (rebuilt each round).
        c_local_dev: Optional[dict] = None
        c_global_dev: Optional[dict] = None
        if self.fed_algo == 'scaffold':
            if self._scaffold_c_local is None:
                self._scaffold_c_local = self._scaffold_zero_dict()
            cg = self._scaffold_c_global if self._scaffold_c_global is not None \
                 else self._scaffold_zero_dict()
            c_local_dev  = {n: t.to(self.device) for n, t in self._scaffold_c_local.items()}
            c_global_dev = {n: t.to(self.device) for n, t in cg.items()}
        # Tracks total local optimizer steps across all epochs in this round.
        K_steps = 0
        # Reset stashes so the orchestrator never reads stale data.
        self.last_scaffold_delta_c = None
        self.last_extra_grad_stats = None
        # Per-step gradient-contribution log (only populated for fedprox/scaffold).
        # task_norms / extra_norms are pre-clip L2 norms over federated params.
        # dots are <g_task, g_extra> (used to compute cosine).
        eg_task: list = []
        eg_extra: list = []
        eg_dot:  list = []

        # Reset MLDG patient-split counters for this round.
        self._mldg_used_steps     = 0
        self._mldg_fallback_steps = 0
        # H1: per-step grad alignment log, populated only when mldg_log_grad_alignment.
        self._mldg_grad_log: list = []
        self._mldg_current_round = int(comm_round_idx) if comm_round_idx is not None else -1
        if self.fed_algo != 'fedavg':
            extra = (f"mu={self.fedprox_mu}" if self.fed_algo == 'fedprox'
                     else f"lr_g={self.scaffold_lr_g}")
            logger.info(f"Federated algorithm: {self.fed_algo} ({extra})")
        if self.mldg:
            inner_lr_str = str(self.mldg_inner_lr) if self.mldg_inner_lr is not None else "same as outer"
            logger.info(
                f"MLDG enabled ({'first-order' if self.mldg_first_order else 'second-order/higher'}) — "
                f"n_support={self.mldg_n_support}, n_query={self.mldg_n_query}, "
                f"inner_iters={self.mldg_inner_iters}, trade_off={self.mldg_trade_off}, "
                f"inner_lr={inner_lr_str}"
            )

        best_val_loss = float('inf')
        best_state    = None
        train_losses, val_losses = [], []
        epoch_len = len(str(self.num_epochs))

        h = self.horizon_30min_idx

        for epoch in range(self.num_epochs):
            # --- training phase ---
            self.model.train()
            ep_train = []
            step_acc = []   # accumulates losses for the current 500-step window
            sq30_sum = 0.0  # sum of squared errors at 30-min horizon (mg/dL^2)
            sq30_n   = 0    # count of samples contributing to sq30_sum
            for step, batch in enumerate(self.train_loader, 1):
                if self.mldg:
                    if self.mldg_first_order:
                        loss_val = self._train_step_mldg_fo(batch, optimizer)
                    elif self.mldg_inner_reuse_outer_opt:
                        loss_val = self._train_step_mldg_ref(batch, optimizer)
                    else:
                        loss_val = self._train_step_mldg(batch, optimizer)
                else:
                    x_enc, x_mark_enc, x_dec, x_mark_dec, y, cgm_mean, cgm_std, _ = \
                        [t.to(self.device) for t in batch]
                    optimizer.zero_grad()
                    out  = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    loss = self.quantile_loss(out, y)
                    # Task-only backward.  FedProx's prox grad and SCAFFOLD's
                    # correction are both closed-form, so we add them
                    # analytically below — and log their magnitudes against
                    # the task gradient for diagnostic purposes.
                    loss.backward()

                    if self.fed_algo in ('fedprox', 'scaffold'):
                        with torch.no_grad():
                            task_sq  = 0.0
                            extra_sq = 0.0
                            dot      = 0.0
                            for n, p in self._federated_named_parameters():
                                if p.grad is None:
                                    continue
                                if self.fed_algo == 'fedprox':
                                    e = self.fedprox_mu * (p.data - x_snapshot[n])
                                else:  # scaffold
                                    e = c_global_dev[n] - c_local_dev[n]
                                task_sq  += float((p.grad ** 2).sum())
                                extra_sq += float((e ** 2).sum())
                                dot      += float((p.grad * e).sum())
                                if self.fed_algo == 'fedprox':
                                    # Apply prox grad before clip — same
                                    # semantics as the old loss-augmentation
                                    # form, but separable for logging.
                                    p.grad.add_(e)
                            eg_task.append(task_sq ** 0.5)
                            eg_extra.append(extra_sq ** 0.5)
                            eg_dot.append(dot)

                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if self.fed_algo == 'scaffold':
                        # SCAFFOLD: g ← g − c_local + c_global.  Applied AFTER
                        # clipping — the correction is a bias, not a gradient.
                        # Only federated params have control variates.
                        for n, p in self._federated_named_parameters():
                            if p.grad is None:
                                continue
                            p.grad.add_(c_global_dev[n] - c_local_dev[n])
                    optimizer.step()
                    K_steps += 1
                    loss_val = loss.item()

                    # Running train RMSE @ 30-min in mg/dL using the same forward pass.
                    # std cancels mean in the difference, so se = ((pred - true) * std)^2.
                    with torch.no_grad():
                        pred_h = self.quantile_loss.median(out)[:, h, 0]
                        true_h = y[:, h, 0]
                        diff_d = (pred_h - true_h) * cgm_std.view(-1)
                        sq30_sum += float((diff_d ** 2).sum().item())
                        sq30_n   += int(diff_d.numel())

                if scheduler is not None:
                    scheduler.step()
                ep_train.append(loss_val)
                step_acc.append(loss_val)

                if step % 500 == 0:
                    logger.info(
                        f"  step {step:>6}  avg-500={float(np.mean(step_acc)):.5f}"
                    )
                    step_acc = []

                if self.max_steps_per_epoch and step >= self.max_steps_per_epoch:
                    break

            t_loss = float(np.mean(ep_train)) if ep_train else float('inf')
            train_losses.append(t_loss)
            train_rmse_30 = (
                float(np.sqrt(sq30_sum / sq30_n)) if sq30_n > 0 else float('nan')
            )

            # --- validation phase (only when early_stopping=True) ---
            v_loss = float('nan')
            if self.early_stopping and self.val_loader is not None:
                self.model.eval()
                ep_val = []
                with torch.no_grad():
                    for batch in self.val_loader:
                        x_enc, x_mark_enc, x_dec, x_mark_dec, y, _, _, _ = \
                            [t.to(self.device) for t in batch]
                        out  = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        loss = self.quantile_loss(out, y)
                        ep_val.append(loss.item())
                v_loss = float(np.mean(ep_val)) if ep_val else float('inf')
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_state    = copy.deepcopy(self.model.state_dict())

            val_losses.append(v_loss)

            rmse_str = (
                f"  train_rmse_30={train_rmse_30:.3f}mg/dL"
                if not math.isnan(train_rmse_30) else ""
            )
            logger.info(
                f"[{epoch+1:>{epoch_len}}/{self.num_epochs}] "
                f"train={t_loss:.5f}"
                + rmse_str
                + (f"  val={v_loss:.5f}" if self.early_stopping else "")
                + f"  lr={optimizer.param_groups[0]['lr']:.2e}"
            )

            if plt is not None:   # skip per-epoch debug plots when matplotlib absent
                self._plot_debug_samples(comm_round_idx, epoch + 1)

        if self.early_stopping and best_state is not None:
            save_state = best_state
            logger.info(f"Training done (early stopping). Best val loss: {best_val_loss:.5f}")
            # Realign live model params with what we're returning, so SCAFFOLD's
            # Δc below is computed against the actually-saved y_i.
            self.model.load_state_dict(best_state)
        else:
            save_state = self.model.state_dict()
            logger.info("Training done (no early stopping — using final epoch weights).")

        # ---- SCAFFOLD: option-II local-c update --------------------------
        # c_i⁺ = c_i − c + (x − y_i) / (K · η_l).  Δc_i = c_i⁺ − c_i is what
        # the server averages.  Persist c_i⁺ on CPU for next round.
        if self.fed_algo == 'scaffold':
            K = max(K_steps, 1)
            eta_l = float(self.learning_rate)
            new_c_local: dict = {}
            delta_c:     dict = {}
            for n, p in self._federated_named_parameters():
                x_p = x_snapshot[n]
                y_p = p.data
                c_old = c_local_dev[n]
                c_new = c_old - c_global_dev[n] + (x_p - y_p) / (K * eta_l)
                delta_c[n]     = (c_new - c_old).detach().cpu()
                new_c_local[n] = c_new.detach().cpu()
            self._scaffold_c_local      = new_c_local
            self.last_scaffold_delta_c  = delta_c
            total_dc = math.sqrt(sum(float(t.pow(2).sum()) for t in delta_c.values()))
            total_cl = math.sqrt(sum(float(t.pow(2).sum()) for t in new_c_local.values()))
            logger.info(
                f"SCAFFOLD: K={K}  ||Δc_i||={total_dc:.4g}  ||c_i⁺||={total_cl:.4g}"
            )

        # ---- Extra-gradient logging summary -----------------------------
        # Records how much of the local gradient comes from the FedProx prox
        # term or the SCAFFOLD correction, vs. the task gradient.  Health rules
        # of thumb: FedProx ratio < 30%, SCAFFOLD ratio in [10%, 100%].
        if eg_task:
            t = np.array(eg_task)
            e = np.array(eg_extra)
            d = np.array(eg_dot)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratios  = np.where(t > 0, e / t, np.nan)
                cosines = np.where((t > 0) & (e > 0), d / (t * e), np.nan)
            self.last_extra_grad_stats = {
                "fed_algo":        self.fed_algo,
                "n_steps":         int(len(t)),
                "task_norm_mean":  float(t.mean()),
                "extra_norm_mean": float(e.mean()),
                "ratio_mean":      float(np.nanmean(ratios)),
                "ratio_p50":       float(np.nanmedian(ratios)),
                "ratio_p90":       float(np.nanpercentile(ratios, 90))
                                   if len(ratios) > 1 else float(ratios[0]),
                "cos_mean":        float(np.nanmean(cosines)),
            }
            logger.info(
                f"Extra grad ({self.fed_algo}): "
                f"||g_task||={t.mean():.4g}  ||g_extra||={e.mean():.4g}  "
                f"ratio={float(np.nanmean(ratios)):.3f} "
                f"(p50={float(np.nanmedian(ratios)):.3f}, "
                f"p90={float(np.nanpercentile(ratios, 90)):.3f})  "
                f"cos={float(np.nanmean(cosines)):+.3f}"
            )

        # Stash per-round MLDG split stats so the orchestrator can persist them.
        if self.mldg:
            used  = self._mldg_used_steps
            drop  = self._mldg_fallback_steps
            total = used + drop
            pct   = (100.0 * used / total) if total else 0.0
            logger.info(
                f"MLDG split: {used}/{total} batches used patient-disjoint split "
                f"({pct:.1f}%); {drop} fell back to vanilla SGD."
            )
            self.last_mldg_stats = {
                "round":    int(comm_round_idx),
                "used":     used,
                "fallback": drop,
                "total":    total,
            }
            # H1: append this round's per-step grad-alignment log to CSV.
            if self.mldg_log_grad_alignment and self._mldg_grad_log:
                out_csv = os.path.join(
                    self.plot_root, f"mldg_grad_align__{self.source_name}.csv"
                )
                os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
                new_file = not os.path.exists(out_csv)
                with open(out_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(self._mldg_grad_log[0].keys()))
                    if new_file:
                        w.writeheader()
                    w.writerows(self._mldg_grad_log)
                cos_vals = [r["cos"] for r in self._mldg_grad_log]
                logger.info(
                    f"MLDG grad-align: round {int(comm_round_idx)}  "
                    f"n_steps={len(cos_vals)}  cos mean={float(np.mean(cos_vals)):+.3f}  "
                    f"median={float(np.median(cos_vals)):+.3f}  "
                    f"appended to {out_csv}"
                )
        else:
            self.last_mldg_stats = None

        # With partial FL we only ship federated keys to the aggregator —
        # client-specific keys never leave the client.
        buf = io.BytesIO()
        torch.save(self._filter_state_dict_federated(save_state), buf)
        return buf.getvalue(), self.n_train, train_losses, val_losses

    # ------------------------------------------------------------------ evaluate

    def evaluate(self, parameters: Optional[bytes] = None) -> float:
        """
        Evaluate the aggregated model on the metabonet_splits val split.
        Returns RMSE (mg/dL) at the 30-min horizon.
        """
        self._load_global_params(parameters, context="evaluate")
        self.model.to(self.device)
        self.model.eval()
        return self._evaluate_loader(self.val_loader, label="val")

    def evaluate_local_val(self, parameters: Optional[bytes] = None) -> dict:
        """
        Evaluate on this client's local validation split.

        Returns a metrics dict containing at least:
          - mse_norm
          - mse_mgdl_30min
          - rmse_mgdl_30min
        """
        self._load_global_params(parameters, context="evaluate_local_val")
        self.model.to(self.device)
        self.model.eval()
        return self._evaluate_loader_metrics(self.val_loader, label="local-val")

    def evaluate_local_test(self, parameters: Optional[bytes] = None) -> dict:
        """
        Evaluate on this client's local test split.

        Returns a metrics dict containing MSE / RMSE at 30-min horizon and
        IRT-style percentages for actual vs predicted glucose ranges.
        """
        self._load_global_params(parameters, context="evaluate_local_test")
        self.model.to(self.device)
        self.model.eval()
        return self._evaluate_loader_metrics(self.test_loader, label="local-test")

    def evaluate_test(self, parameters: Optional[bytes] = None) -> float:
        """
        Evaluate on the metabonet_splits test split.
        Returns RMSE (mg/dL) at the 30-min horizon.
        """
        self._load_global_params(parameters, context="evaluate_test")
        self.model.to(self.device)
        self.model.eval()
        return self._evaluate_loader(self.test_loader, label="test")

    # ------------------------------------------------------------------ OOD fine-tune

    def finetune_ood(
        self,
        num_epochs: int = 10,
        lr: Optional[float] = None,
        finetune_params: str = "non_federated",
        max_steps_per_epoch: Optional[int] = None,
    ) -> dict:
        """Fine-tune the model on this client's local train set, val-best selected.

        Used for OOD eval under partial FL: starting from
        (federated encoder + averaged decoder), train on self.train_loader
        for `num_epochs` epochs, picking the epoch with the lowest quantile
        loss on self.val_loader.

        Args:
          num_epochs: epochs over self.train_loader.
          lr: learning rate (None → self.learning_rate).
          finetune_params: "non_federated" (decoder only — default), "federated"
            (encoder only), or "all".
          max_steps_per_epoch: cap (None → self.max_steps_per_epoch).

        Returns:
          dict with best_epoch, best_val_loss, train_losses, val_losses,
          n_trainable_params.

        Side effects: self.model is set to the best-val state at return.
        """
        if num_epochs <= 0 or self.train_loader is None or self.val_loader is None:
            logger.warning(
                f"finetune_ood({self.source_name}): "
                f"num_epochs={num_epochs}, train_loader={self.train_loader is not None}, "
                f"val_loader={self.val_loader is not None} — skipping"
            )
            return {"best_epoch": -1, "best_val_loss": float("nan"),
                    "train_losses": [], "val_losses": [], "n_trainable_params": 0}

        fed_set = bool(self.federated_param_prefixes)

        def _train_this(name: str) -> bool:
            is_fed = self._is_federated_key(name)
            if finetune_params == "all":
                return True
            if finetune_params == "federated":
                return is_fed
            if finetune_params == "non_federated":
                return (not is_fed) if fed_set else False
            raise ValueError(
                f"finetune_ood: unknown finetune_params='{finetune_params}'")

        # Save requires_grad state, then restrict to the selected subset.
        prev_rg, selectable = {}, []
        for n, p in self.model.named_parameters():
            prev_rg[n] = p.requires_grad
            p.requires_grad_(prev_rg[n] and _train_this(n))
            if p.requires_grad:
                selectable.append(p)

        if not selectable:
            for n, p in self.model.named_parameters():
                p.requires_grad_(prev_rg[n])
            logger.warning(
                f"finetune_ood({self.source_name}): no params selected "
                f"(finetune_params={finetune_params}, "
                f"federated_param_prefixes={self.federated_param_prefixes}) — skipping"
            )
            return {"best_epoch": -1, "best_val_loss": float("nan"),
                    "train_losses": [], "val_losses": [], "n_trainable_params": 0}

        eta       = float(lr) if lr is not None else float(self.learning_rate)
        steps_cap = (max_steps_per_epoch
                     if max_steps_per_epoch is not None
                     else self.max_steps_per_epoch)
        self.model.to(self.device)
        optimizer = optim.AdamW(selectable, lr=eta, weight_decay=self.weight_decay)

        best_val   = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        train_losses, val_losses = [], []
        n_trainable = sum(p.numel() for p in selectable)

        logger.info(
            f"OOD finetune ({self.source_name}): params={finetune_params} "
            f"({n_trainable} trainable), epochs={num_epochs}, lr={eta:.2e}"
        )

        try:
            for epoch in range(1, num_epochs + 1):
                self.model.train()
                ep_train = []
                for step, batch in enumerate(self.train_loader, 1):
                    x_enc, x_mark_enc, x_dec, x_mark_dec, y, _, _, _ = \
                        [t.to(self.device) for t in batch]
                    optimizer.zero_grad()
                    out  = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    loss = self.quantile_loss(out, y)
                    loss.backward()
                    nn.utils.clip_grad_norm_(selectable, 1.0)
                    optimizer.step()
                    ep_train.append(loss.item())
                    if steps_cap and step >= steps_cap:
                        break
                t_loss = float(np.mean(ep_train)) if ep_train else float("inf")
                train_losses.append(t_loss)

                self.model.eval()
                ep_val = []
                with torch.no_grad():
                    for batch in self.val_loader:
                        x_enc, x_mark_enc, x_dec, x_mark_dec, y, _, _, _ = \
                            [t.to(self.device) for t in batch]
                        out  = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        ep_val.append(self.quantile_loss(out, y).item())
                v_loss = float(np.mean(ep_val)) if ep_val else float("inf")
                val_losses.append(v_loss)

                improved = v_loss < best_val
                if improved:
                    best_val   = v_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch
                logger.info(
                    f"  finetune ep {epoch}/{num_epochs}: "
                    f"train={t_loss:.5f}  val={v_loss:.5f}"
                    + ("  *new best*" if improved else "")
                )
        finally:
            for n, p in self.model.named_parameters():
                p.requires_grad_(prev_rg[n])

        self.model.load_state_dict(best_state)
        return {
            "best_epoch":         best_epoch,
            "best_val_loss":      best_val,
            "train_losses":       train_losses,
            "val_losses":         val_losses,
            "n_trainable_params": int(n_trainable),
        }

    # ------------------------------------------------------------------ helpers

    def _evaluate_loader(self, loader, label: str = "test") -> float:
        """Evaluate on a DataLoader (internal fallback)."""
        if loader is None:
            logger.warning(f"No {label} loader — returning inf")
            return float('inf')

        h = self.horizon_30min_idx
        sq_errors = []

        with torch.no_grad():
            for batch in loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, cgm_mean, cgm_std, _ = \
                    [t.to(self.device) for t in batch]
                out       = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                pred_norm = self.quantile_loss.median(out)
                pred_h    = pred_norm[:, h, 0]
                true_h    = y[:, h, 0]
                mean      = cgm_mean.view(-1)
                std       = cgm_std.view(-1)
                pred_d    = pred_h * std + mean
                true_d    = true_h * std + mean
                sq_errors.append(((pred_d - true_d) ** 2).cpu().numpy())

        if not sq_errors:
            return float('inf')

        rmse = float(np.sqrt(np.mean(np.concatenate(sq_errors))))
        logger.info(f"{label} RMSE at 30-min horizon: {rmse:.3f} mg/dL")
        return rmse

    def _evaluate_loader_metrics(self, loader, label: str) -> dict:
        """
        Evaluate a local split loader and return MSE/RMSE + IRT-style metrics.
        """
        if loader is None:
            logger.warning(f"No {label} loader — returning empty metrics")
            return {
                "mse_norm": float("inf"),
                "mse_mgdl_30min": float("inf"),
                "rmse_mgdl_30min": float("inf"),
                "rmse_mgdl_60min": float("nan"),
                "rmse_mgdl_120min": float("nan"),
                "time_lag_30min_min": float("nan"),
                "pinball_30min_mgdl_avg":  float("nan"),
                "pinball_30min_mgdl_q10":  float("nan"),
                "pinball_30min_mgdl_q25":  float("nan"),
                "pinball_30min_mgdl_q50":  float("nan"),
                "pinball_30min_mgdl_q75":  float("nan"),
                "pinball_30min_mgdl_q90":  float("nan"),
                "pi80_coverage_30min":     float("nan"),
                "pi80_width_30min_mgdl":   float("nan"),
                "pi50_coverage_30min":     float("nan"),
                "pi50_width_30min_mgdl":   float("nan"),
                "tir_actual": float("nan"),
                "tir_predicted": float("nan"),
                "tir_difference": float("nan"),
                "tir_hypo_actual": float("nan"),
                "tir_hypo_predicted": float("nan"),
                "tir_hyper_actual": float("nan"),
                "tir_hyper_predicted": float("nan"),
            }

        h = self.horizon_30min_idx
        h60 = 11
        h120 = 23
        sq_norm_sum = 0.0
        n_norm = 0
        sq_30 = []
        sq_60 = []
        sq_120 = []
        true_30 = []
        pred_30 = []

        # Per-quantile collection at h=30min — only meaningful when the model
        # has multiple quantile heads (skipped under MSELossAdapter where
        # quantiles == [0.5] and c_out == 1).
        quantiles = list(self.quantile_loss.quantiles)
        is_quantile_model = len(quantiles) > 1 and label == "local-test"
        q_pred_30_mgdl = [[] for _ in quantiles] if is_quantile_model else None
        q_true_30_mgdl = []        if is_quantile_model else None

        with torch.no_grad():
            for batch in loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, cgm_mean, cgm_std, _ = \
                    [t.to(self.device) for t in batch]
                out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                pred_norm = self.quantile_loss.median(out)

                # mse_norm is the model-selection metric: masked to real
                # (non-interpolated) targets when y carries the cgm_real mask.
                if y.shape[-1] > 1:
                    msk = y[..., 1:2]
                    se_norm = ((pred_norm - y[..., :1]) ** 2) * msk
                    sq_norm_sum += float(se_norm.sum().item())
                    n_norm += int(msk.sum().item())
                else:
                    se_norm = (pred_norm - y) ** 2
                    sq_norm_sum += float(se_norm.sum().item())
                    n_norm += int(se_norm.numel())

                pred_h = pred_norm[:, h, 0]
                true_h = y[:, h, 0]
                mean = cgm_mean.view(-1)
                std = cgm_std.view(-1)

                pred_d = (pred_h * std + mean).detach().cpu().numpy()
                true_d = (true_h * std + mean).detach().cpu().numpy()
                pred_30.append(pred_d)
                true_30.append(true_d)
                sq_30.append((pred_d - true_d) ** 2)

                pred_len = pred_norm.shape[1]
                if h60 < pred_len:
                    pred_h60 = pred_norm[:, h60, 0]
                    true_h60 = y[:, h60, 0]
                    pred_d60 = (pred_h60 * std + mean).detach().cpu().numpy()
                    true_d60 = (true_h60 * std + mean).detach().cpu().numpy()
                    sq_60.append((pred_d60 - true_d60) ** 2)

                if h120 < pred_len:
                    pred_h120 = pred_norm[:, h120, 0]
                    true_h120 = y[:, h120, 0]
                    pred_d120 = (pred_h120 * std + mean).detach().cpu().numpy()
                    true_d120 = (true_h120 * std + mean).detach().cpu().numpy()
                    sq_120.append((pred_d120 - true_d120) ** 2)

                # Test-time only: collect ALL quantile predictions at h=30min
                # in mg/dL.  `out` has shape (B, pred_len, n_quantiles); the
                # quantile axis matches `self.quantile_loss.quantiles`.
                if is_quantile_model:
                    # (B, n_quantiles) at h=30
                    q_pred_h_norm = out[:, h, :]
                    q_pred_h_mgdl = (
                        q_pred_h_norm * std.unsqueeze(-1) + mean.unsqueeze(-1)
                    ).detach().cpu().numpy()       # (B, n_quantiles)
                    for qi in range(len(quantiles)):
                        q_pred_30_mgdl[qi].append(q_pred_h_mgdl[:, qi])
                    q_true_30_mgdl.append(true_d)

        if not sq_30 or n_norm <= 0:
            logger.warning(f"No valid batches in {label} — returning inf metrics")
            return {
                "mse_norm": float("inf"),
                "mse_mgdl_30min": float("inf"),
                "rmse_mgdl_30min": float("inf"),
                "rmse_mgdl_60min": float("nan"),
                "rmse_mgdl_120min": float("nan"),
                "time_lag_30min_min": float("nan"),
                "pinball_30min_mgdl_avg":  float("nan"),
                "pinball_30min_mgdl_q10":  float("nan"),
                "pinball_30min_mgdl_q25":  float("nan"),
                "pinball_30min_mgdl_q50":  float("nan"),
                "pinball_30min_mgdl_q75":  float("nan"),
                "pinball_30min_mgdl_q90":  float("nan"),
                "pi80_coverage_30min":     float("nan"),
                "pi80_width_30min_mgdl":   float("nan"),
                "pi50_coverage_30min":     float("nan"),
                "pi50_width_30min_mgdl":   float("nan"),
                "tir_actual": float("nan"),
                "tir_predicted": float("nan"),
                "tir_difference": float("nan"),
                "tir_hypo_actual": float("nan"),
                "tir_hypo_predicted": float("nan"),
                "tir_hyper_actual": float("nan"),
                "tir_hyper_predicted": float("nan"),
            }

        sq_30_np = np.concatenate(sq_30)
        true_30_np = np.concatenate(true_30)
        pred_30_np = np.concatenate(pred_30)

        normal_low = 70.0
        normal_high = 140.0
        hypo_threshold = 70.0
        hyper_threshold = 180.0

        tir_actual = float(np.mean((true_30_np >= normal_low) & (true_30_np <= normal_high)) * 100.0)
        tir_pred = float(np.mean((pred_30_np >= normal_low) & (pred_30_np <= normal_high)) * 100.0)
        tir_hypo_actual = float(np.mean(true_30_np < hypo_threshold) * 100.0)
        tir_hypo_pred = float(np.mean(pred_30_np < hypo_threshold) * 100.0)
        tir_hyper_actual = float(np.mean(true_30_np > hyper_threshold) * 100.0)
        tir_hyper_pred = float(np.mean(pred_30_np > hyper_threshold) * 100.0)

        # Time-lag (test-time only): cross-correlation lag of the 30-min-ahead
        # predicted CGM series against the truth, stitched across contiguous
        # test-window runs.  Sign convention: positive lag = pred is delayed
        # (a "trivial copy-current-CGM" predictor → +30 min).
        time_lag_30min_min = float("nan")
        if label == "local-test":
            try:
                idx = np.asarray(loader.dataset.index, dtype=np.int64)
                if idx.shape[0] == len(pred_30_np) and idx.shape[1] >= 2:
                    time_lag_30min_min = self._compute_time_lag_from_windows(
                        pred_30_np, true_30_np, idx,
                    )
            except Exception as e:
                logger.warning(f"time-lag computation failed: {e}")

        # ── Quantile metrics at h=30min (test-time only; multi-quantile only)
        # All in mg/dL.  Quantile predictions are NOT sorted defensively —
        # crossing (q0.9 < q0.1 on some windows) shows up as low coverage /
        # tiny / negative width, which is the correct miscalibration signal.
        per_q_pinball_30 = {q: float("nan") for q in quantiles}
        pinball_30_avg   = float("nan")
        pi80_cov_30      = float("nan")
        pi50_cov_30      = float("nan")
        pi80_width_30    = float("nan")
        pi50_width_30    = float("nan")
        if is_quantile_model and q_true_30_mgdl:
            q_true = np.concatenate(q_true_30_mgdl)   # (N,)
            q_pred = np.stack(
                [np.concatenate(per_q) for per_q in q_pred_30_mgdl],
                axis=1,
            )                                          # (N, n_quantiles)
            # Pinball: ρ_q(e) = max((q-1)·e, q·e), e = true − pred_q.
            for qi, q in enumerate(quantiles):
                e = q_true - q_pred[:, qi]
                pin = np.mean(np.maximum((q - 1.0) * e, q * e))
                per_q_pinball_30[q] = float(pin)
            pinball_30_avg = float(np.mean(list(per_q_pinball_30.values())))

            # Coverage + width.  Build interval for each requested PI by
            # picking the closest available quantile pair.
            q_arr = np.asarray(quantiles)
            def _coverage_and_width(low_q, high_q):
                if low_q not in quantiles or high_q not in quantiles:
                    return float("nan"), float("nan")
                lo_idx = quantiles.index(low_q)
                hi_idx = quantiles.index(high_q)
                lo = q_pred[:, lo_idx]
                hi = q_pred[:, hi_idx]
                cov = float(np.mean((q_true >= lo) & (q_true <= hi)))
                wid = float(np.mean(hi - lo))
                return cov, wid

            pi80_cov_30, pi80_width_30 = _coverage_and_width(0.1, 0.9)
            pi50_cov_30, pi50_width_30 = _coverage_and_width(0.25, 0.75)

        metrics = {
            "mse_norm": float(sq_norm_sum / max(n_norm, 1)),
            "mse_mgdl_30min": float(np.mean(sq_30_np)),
            "rmse_mgdl_30min": float(np.sqrt(np.mean(sq_30_np))),
            "rmse_mgdl_60min": float(np.sqrt(np.mean(np.concatenate(sq_60)))) if sq_60 else float("nan"),
            "rmse_mgdl_120min": float(np.sqrt(np.mean(np.concatenate(sq_120)))) if sq_120 else float("nan"),
            "time_lag_30min_min": time_lag_30min_min,
            "pinball_30min_mgdl_avg":  pinball_30_avg,
            "pinball_30min_mgdl_q10":  per_q_pinball_30.get(0.1,  float("nan")),
            "pinball_30min_mgdl_q25":  per_q_pinball_30.get(0.25, float("nan")),
            "pinball_30min_mgdl_q50":  per_q_pinball_30.get(0.5,  float("nan")),
            "pinball_30min_mgdl_q75":  per_q_pinball_30.get(0.75, float("nan")),
            "pinball_30min_mgdl_q90":  per_q_pinball_30.get(0.9,  float("nan")),
            "pi80_coverage_30min":     pi80_cov_30,
            "pi80_width_30min_mgdl":   pi80_width_30,
            "pi50_coverage_30min":     pi50_cov_30,
            "pi50_width_30min_mgdl":   pi50_width_30,
            "tir_actual": tir_actual,
            "tir_predicted": tir_pred,
            "tir_difference": float(abs(tir_actual - tir_pred)),
            "tir_hypo_actual": tir_hypo_actual,
            "tir_hypo_predicted": tir_hypo_pred,
            "tir_hyper_actual": tir_hyper_actual,
            "tir_hyper_predicted": tir_hyper_pred,
        }

        tl_str = (f" TL={time_lag_30min_min:+.1f}min"
                  if not np.isnan(time_lag_30min_min) else "")
        quant_str = ""
        if not np.isnan(pinball_30_avg):
            quant_str = (
                f" pinball_30(avg)={pinball_30_avg:.3f}"
                f" PI80(cov={pi80_cov_30*100:.1f}%, w={pi80_width_30:.1f}mg/dL)"
                f" PI50(cov={pi50_cov_30*100:.1f}%, w={pi50_width_30:.1f}mg/dL)"
            )
        logger.info(
            f"{label} metrics: mse_norm={metrics['mse_norm']:.6f} "
            f"mse_30={metrics['mse_mgdl_30min']:.3f} "
            f"rmse_30={metrics['rmse_mgdl_30min']:.3f} "
            f"rmse_60={metrics['rmse_mgdl_60min']:.3f} "
            f"rmse_120={metrics['rmse_mgdl_120min']:.3f}"
            f"{tl_str}"
            f"{quant_str} "
            f"TIR(actual/pred)={metrics['tir_actual']:.1f}%/{metrics['tir_predicted']:.1f}%"
        )
        return metrics

    # ------------------------------------------------------------------ time lag

    @staticmethod
    def _cross_corr_lag(pred: np.ndarray, true: np.ndarray, K: int) -> float:
        """
        Lag (in SAMPLES) that maximises corr(true[t], pred[t+τ]) for τ ∈ [-K, K].
        Sign convention: positive lag = prediction is delayed.

        Returns NaN if series is too short or has zero variance.
        """
        n = len(pred)
        if n < 2 * K + 4:
            return float("nan")
        p = pred - pred.mean()
        t = true - true.mean()
        p_norm = float(np.linalg.norm(p))
        t_norm = float(np.linalg.norm(t))
        if p_norm < 1e-12 or t_norm < 1e-12:
            return float("nan")
        best_tau, best_corr = 0, -float("inf")
        for tau in range(-K, K + 1):
            if tau >= 0:
                ps = p[tau:]
                ts = t[: n - tau]
            else:
                ps = p[: n + tau]
                ts = t[-tau:]
            L = min(len(ps), len(ts))
            if L < 4:
                continue
            corr = float(np.dot(ps[:L], ts[:L])) / (p_norm * t_norm + 1e-12)
            if corr > best_corr:
                best_corr = corr
                best_tau = tau
        return float(best_tau)

    def _compute_time_lag_from_windows(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        win_idx: np.ndarray,
        max_lag_min: int = 60,
        min_run_len: int = 12,
    ) -> float:
        """
        Stitch contiguous stride-1 runs of test windows and compute a
        length-weighted average cross-correlation lag.

        win_idx columns: [seg_idx, start_pos, (optional) pat_idx].
        A "run" is a maximal stretch where (pat, seg) is constant and
        start_pos increments by exactly 1.  Runs shorter than `min_run_len
        + 2*K` (K = max_lag_min // sampling_min) are skipped.

        Returns lag in MINUTES (positive = pred delayed), or NaN.
        """
        sampling_min = int(getattr(self._data_cfg, "sampling_minutes", 5))
        K = int(max_lag_min // sampling_min)

        seg   = win_idx[:, 0]
        start = win_idx[:, 1]
        pat   = win_idx[:, 2] if win_idx.shape[1] >= 3 else np.full_like(seg, -1)

        order = np.lexsort((start, seg, pat))
        pred_o  = pred[order]
        true_o  = true[order]
        seg_o   = seg[order]
        start_o = start[order]
        pat_o   = pat[order]

        N = len(pred_o)
        total_len = 0
        weighted_tau_samples = 0.0
        i = 0
        while i < N:
            j = i + 1
            while (j < N
                   and pat_o[j]  == pat_o[i]
                   and seg_o[j]  == seg_o[i]
                   and start_o[j] == start_o[j - 1] + 1):
                j += 1
            run_len = j - i
            if run_len >= min_run_len + 2 * K:
                tl = self._cross_corr_lag(pred_o[i:j], true_o[i:j], K)
                if not np.isnan(tl):
                    total_len += run_len
                    weighted_tau_samples += tl * run_len
            i = j

        if total_len == 0:
            return float("nan")
        return float(weighted_tau_samples / total_len) * sampling_min

    # ------------------------------------------------------------------ aggregate

    def aggregate(self, parameters_list: List[bytes],
                  sample_counts: List[int],
                  comm_round_idx: Optional[int] = None) -> bytes:
        """
        Federated averaging (FedAvg) weighted by number of training samples.

        Args:
            parameters_list : list of serialised state_dicts from clients
            sample_counts   : number of training samples for each client
            comm_round_idx  : federation round (informational)

        Returns:
            Serialised averaged state_dict.
        """
        if not parameters_list:
            logger.warning("aggregate: empty list — returning current model")
            buf = io.BytesIO()
            torch.save(self._filter_state_dict_federated(self.model.state_dict()), buf)
            return buf.getvalue()

        if len(parameters_list) == 1:
            return parameters_list[0]

        total_samples = sum(sample_counts)
        if total_samples == 0:
            total_samples = len(parameters_list)
            sample_counts = [1] * len(parameters_list)

        models, weights = [], []
        for i, params in enumerate(parameters_list):
            try:
                sd = torch.load(io.BytesIO(params), weights_only=True)
                models.append(sd)
                weights.append(sample_counts[i] / total_samples)
            except Exception as e:
                logger.warning(f"aggregate: skipping client {i}: {e}")

        if not models:
            logger.error("aggregate: no valid models — returning current state")
            buf = io.BytesIO()
            torch.save(self._filter_state_dict_federated(self.model.state_dict()), buf)
            return buf.getvalue()

        agg = {}
        for key in models[0]:
            agg[key] = sum(m[key] * w for m, w in zip(models, weights))

        buf = io.BytesIO()
        torch.save(agg, buf)
        logger.info(f"Aggregated {len(models)} clients (FedAvg)")
        return buf.getvalue()

    # ------------------------------------------------------------------ SCAFFOLD broker

    def _scaffold_zero_dict(self) -> dict:
        """Init a c-dict of zeros shaped like the FEDERATED trainable params (CPU)."""
        return {
            n: torch.zeros_like(p, device='cpu')
            for n, p in self._federated_named_parameters()
        }

    def set_scaffold_c_global(self, c_global: Optional[dict]) -> None:
        """Server -> client broadcast of the global control variate.

        Pass None on round 1 (server has no c yet); the client lazy-inits to zeros.
        """
        self._scaffold_c_global = c_global

    def aggregate_scaffold_c(self, delta_c_list, sample_counts) -> None:
        """Update server-side c_global ← c_global + Σ w_i Δc_i (sample-weighted).

        Called once per round on the aggregator client, AFTER aggregate().
        Equivalent to standard SCAFFOLD with full participation (|S|=N).
        """
        valid = [(d, s) for d, s in zip(delta_c_list, sample_counts) if d is not None]
        if not valid:
            return
        deltas, counts = zip(*valid)
        total = float(sum(counts)) or 1.0
        weights = [s / total for s in counts]
        if self._scaffold_c_global is None:
            self._scaffold_c_global = {n: torch.zeros_like(t) for n, t in deltas[0].items()}
        for name in self._scaffold_c_global:
            delta_avg = sum(dc[name] * w for dc, w in zip(deltas, weights))
            self._scaffold_c_global[name] = self._scaffold_c_global[name] + delta_avg

    # ------------------------------------------------------------------ debug plots

    def _plot_debug_samples(
        self,
        round_idx: int,
        epoch:     int,
        n:         int = 5,
    ) -> None:
        """
        Save a 2-row × n-col debug plot after each training epoch.

        Row 0: n samples from the training loader (model in eval mode).
        Row 1: n samples from the validation loader.
        Context in blue, target in green, median prediction in red,
        q10–q90 band shaded. Values in mg/dL.

        Saved to: <plot_root>/debug/<source_name>/r<round>_e<epoch>.png
        """
        source_name = os.path.basename(self.source_dir)
        out_dir     = os.path.join(self.plot_root, "debug", source_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"r{round_idx:03d}_e{epoch:03d}.png")

        cfg        = self._data_cfg
        step_min   = cfg.sampling_minutes
        ctx_t      = np.arange(-cfg.seq_len,  0) * step_min
        pred_t     = np.arange(1, cfg.pred_len + 1) * step_min
        # Adapt to whichever output head we have: quantile (5-head: q10..q90)
        # or MSE single-head (c_out=1).  Under MSE there's no band to shade.
        quantiles = list(self.quantile_loss.quantiles)
        if len(quantiles) >= 5 and {0.1, 0.5, 0.9}.issubset(set(quantiles)):
            q10_idx = quantiles.index(0.1)
            q50_idx = quantiles.index(0.5)
            q90_idx = quantiles.index(0.9)
            has_band = True
        else:
            q10_idx = q50_idx = q90_idx = 0
            has_band = False

        def _collect(loader, n_samples):
            samples = []
            for batch in loader:
                xe, xme, xd, xmd, y, g_mean_b, g_std_b, _ = batch
                for i in range(xe.size(0)):
                    samples.append((
                        xe[i], xme[i], xd[i], xmd[i], y[i],
                        float(g_mean_b[i]), float(g_std_b[i]),
                    ))
                    if len(samples) >= n_samples:
                        return samples
            return samples

        self.model.eval()
        with torch.no_grad():
            train_samples = _collect(self.train_loader, n)
            val_samples   = _collect(self.val_loader, n) if self.val_loader is not None else []

        n_rows = 2
        n_cols = max(len(train_samples), len(val_samples), 1)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 3.5 * n_rows),
            squeeze=False,
        )

        row_data   = [("train", train_samples), ("val", val_samples)]
        row_colors = [("steelblue", "seagreen"), ("royalblue", "mediumseagreen")]

        with torch.no_grad():
            for row, ((split_name, samples), (ctx_col, tgt_col)) in enumerate(
                zip(row_data, row_colors)
            ):
                for col in range(n_cols):
                    ax = axes[row][col]
                    if col >= len(samples):
                        ax.axis("off")
                        continue

                    xe, xme, xd, xmd, y_s, g_mean, g_std = samples[col]
                    out    = self.model(
                        xe.unsqueeze(0).to(self.device),
                        xme.unsqueeze(0).to(self.device),
                        xd.unsqueeze(0).to(self.device),
                        xmd.unsqueeze(0).to(self.device),
                    )
                    out_np = out[0].cpu().numpy()          # (pred_len, c_out)

                    ctx_mg  = xe[:, 0].cpu().numpy()  * g_std + g_mean
                    tgt_mg  = y_s[:, 0].cpu().numpy() * g_std + g_mean
                    med_mg  = out_np[:, q50_idx] * g_std + g_mean

                    ax.plot(ctx_t,  ctx_mg, color=ctx_col, lw=1.2, label="context")
                    ax.plot(pred_t, tgt_mg, color=tgt_col, lw=1.2, label="target")
                    ax.plot(pred_t, med_mg, color="firebrick", lw=1.2, ls="--", label="pred")
                    if has_band:
                        q10_mg = out_np[:, q10_idx] * g_std + g_mean
                        q90_mg = out_np[:, q90_idx] * g_std + g_mean
                        ax.fill_between(pred_t, q10_mg, q90_mg,
                                        color="firebrick", alpha=0.12)
                    ax.axvline(0, color="grey", lw=0.7, ls=":")
                    ax.set_xlim(ctx_t[0], pred_t[-1])
                    ax.set_ylabel("mg/dL", fontsize=7)
                    ax.tick_params(labelsize=6)
                    if row == 0:
                        ax.set_title(f"train {col+1}", fontsize=8)
                    else:
                        ax.set_title(f"val {col+1}", fontsize=8)
                    if row == 0 and col == 0:
                        ax.legend(fontsize=6, loc="upper left")
                    ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"{source_name}  round {round_idx}  epoch {epoch}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        logger.debug(f"Debug plot saved: {out_path}")

    # ------------------------------------------------------------------ prediction plots

    def plot_predictions(
        self,
        parameters:   Optional[bytes],
        round_num:    int,
        output_dir:   str,
        n_per_source: int = 2,
    ) -> None:
        """
        Save a grid of context / prediction / target plots from this client's val set.

        1 row × n_per_source columns.
        Context shown in blue, target future in green,
        median prediction in red, q10–q90 band as a shaded area.
        All values are de-normalised to mg/dL.

        Saved to: <output_dir>/predictions_round_<round_num>.png
        """
        if self.val_loader is None:
            logger.warning("plot_predictions: val_loader not available — skipping")
            return

        self._load_global_params(parameters, context="plot_predictions")
        self.model.to(self.device)
        self.model.eval()

        # Collect n_per_source individual samples from the val loader
        samples = []
        for batch in self.val_loader:
            x_enc_b, x_mark_enc_b, x_dec_b, x_mark_dec_b, y_b, cgm_mean_b, cgm_std_b, _ = batch
            for i in range(x_enc_b.size(0)):
                samples.append((
                    x_enc_b[i], x_mark_enc_b[i], x_dec_b[i], x_mark_dec_b[i],
                    y_b[i], float(cgm_mean_b[i]), float(cgm_std_b[i]),
                ))
                if len(samples) >= n_per_source:
                    break
            if len(samples) >= n_per_source:
                break

        if not samples:
            logger.warning("plot_predictions: val_loader yielded no samples — skipping")
            return

        source_name = os.path.basename(self.source_dir)
        n_cols      = len(samples)

        cfg        = self._data_cfg
        step_min   = cfg.sampling_minutes
        ctx_steps  = cfg.seq_len
        pred_steps = cfg.pred_len
        ctx_t      = np.arange(-ctx_steps, 0) * step_min
        pred_t     = np.arange(1, pred_steps + 1) * step_min

        # Adapt to quantile (5-head) or MSE (single-head) output.
        quantiles = list(self.quantile_loss.quantiles)
        if len(quantiles) >= 5 and {0.1, 0.5, 0.9}.issubset(set(quantiles)):
            q10_idx = quantiles.index(0.1)
            q50_idx = quantiles.index(0.5)
            q90_idx = quantiles.index(0.9)
            has_band = True
        else:
            q10_idx = q50_idx = q90_idx = 0
            has_band = False

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(6 * n_cols, 3.5),
            squeeze=False,
        )

        with torch.no_grad():
            for col, (x_enc_s, x_mark_enc_s, x_dec_s, x_mark_dec_s, y_s, g_mean, g_std) in enumerate(samples):
                ax = axes[0][col]

                out    = self.model(
                    x_enc_s.unsqueeze(0).to(self.device),
                    x_mark_enc_s.unsqueeze(0).to(self.device),
                    x_dec_s.unsqueeze(0).to(self.device),
                    x_mark_dec_s.unsqueeze(0).to(self.device),
                )
                out_np = out[0].cpu().numpy()   # (pred_len, c_out)

                ctx_denorm  = x_enc_s[:, 0].cpu().numpy() * g_std + g_mean
                tgt_denorm  = y_s[:, 0].cpu().numpy()     * g_std + g_mean
                pred_median = out_np[:, q50_idx] * g_std + g_mean

                ax.plot(ctx_t,  ctx_denorm,  color='steelblue', lw=1.5, label='Context')
                ax.plot(pred_t, tgt_denorm,  color='seagreen',  lw=1.5, label='Target')
                ax.plot(pred_t, pred_median, color='firebrick', lw=1.5,
                        ls='--', label='Pred (median)')
                if has_band:
                    pred_q10 = out_np[:, q10_idx] * g_std + g_mean
                    pred_q90 = out_np[:, q90_idx] * g_std + g_mean
                    ax.fill_between(pred_t, pred_q10, pred_q90,
                                    color='firebrick', alpha=0.15, label='q10–q90')

                ax.axvline(0, color='grey', lw=0.8, ls=':')
                ax.set_xlim(ctx_t[0], pred_t[-1])
                ax.set_xlabel("Time (min)")
                ax.set_ylabel("CGM (mg/dL)")
                ax.set_title(f"{source_name} — example {col + 1}")
                if col == 0:
                    ax.legend(fontsize=7, loc='upper left')
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"CGMTransformer FL — Round {round_num} predictions", fontsize=13)
        fig.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"predictions_round_{round_num:03d}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info(f"Saved prediction plot: {path}")

    # ------------------------------------------------------------------ MLDG

    def _split_by_patient(
        self,
        x_enc, x_mark_enc, x_dec, x_mark_dec, y, pat_idx,
    ):
        """
        Group a batch into per-patient domains, then assign whole patients to
        support and query so the two sets are patient-disjoint.

        Returns:
            (support, query) where each is a list of 5-tuples
              (x_enc_d, x_mark_enc_d, x_dec_d, x_mark_dec_d, y_d).
            Both lists are non-empty.

            None if the batch contains fewer than 2 distinct patients with
            >= 2 windows each (caller should fall back to a vanilla step).
        """
        pat_np  = pat_idx.detach().cpu().numpy()
        domains = []
        for pi in np.unique(pat_np):
            mask = torch.from_numpy(pat_np == pi).to(x_enc.device)
            if int(mask.sum().item()) < 2:
                continue
            domains.append((
                x_enc     [mask],
                x_mark_enc[mask],
                x_dec     [mask],
                x_mark_dec[mask],
                y         [mask],
            ))

        if len(domains) < 2:
            return None

        random.shuffle(domains)
        total           = sum(d[0].shape[0] for d in domains)
        target_sup_frac = self.mldg_n_support / (self.mldg_n_support + self.mldg_n_query)
        target_sup_wins = total * target_sup_frac

        support, query, sup_count = [], [], 0
        for domain in domains:
            if sup_count < target_sup_wins:
                support.append(domain)
                sup_count += domain[0].shape[0]
            else:
                query.append(domain)

        # Guarantee both sides are non-empty (can happen with two highly
        # imbalanced patients).
        if not support:
            support.append(query.pop())
        if not query:
            query.append(support.pop())
        return support, query

    def _vanilla_step(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y, optimizer) -> float:
        """Standard SGD step on a whole batch — used as MLDG fallback."""
        optimizer.zero_grad()
        out  = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = self.quantile_loss(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def _train_step_mldg_fo(self, batch, optimizer) -> float:
        """
        First-order MLDG with patient-disjoint support/query split.

        Support gradients drive a manual inner update; query gradients are
        computed on the updated params; both are combined and applied via the
        outer optimizer (no graph through the inner step).
        """
        x_enc, x_mark_enc, x_dec, x_mark_dec, y, _, _, pat_idx = \
            [t.to(self.device) for t in batch]

        split = self._split_by_patient(
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, pat_idx
        )
        if split is None:
            self._mldg_fallback_steps += 1
            return self._vanilla_step(
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, optimizer
            )
        self._mldg_used_steps += 1
        support, query = split
        n_sup, n_qry   = len(support), len(query)

        params   = list(self.model.parameters())
        inner_lr = self.mldg_inner_lr if self.mldg_inner_lr is not None else self.learning_rate

        # ── Support loss + gradients (no graph) ──────────────────────────
        sup_loss  = sum(self.quantile_loss(
                        self.model(xe, xme, xde, xdme), ye) / n_sup
                    for xe, xme, xde, xdme, ye in support)
        sup_grads = torch.autograd.grad(sup_loss, params,
                                        create_graph=False, allow_unused=True)
        sup_grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(sup_grads, params)]

        # ── Manual inner update (in-place, no graph) ─────────────────────
        # `mldg_inner_iters` genuine GD steps on the support loss.  The
        # support gradient must be recomputed on the updated params at each
        # step — reusing the stale ∇F(Θ₀) for every iteration would just be
        # one step at inner_iters × inner_lr, not a real multi-step loop.
        # `sup_grads` itself stays = ∇F(Θ₀): that is the outer-update term.
        snapshots  = [p.data.clone() for p in params]
        step_grads = sup_grads
        for _it in range(self.mldg_inner_iters):
            with torch.no_grad():
                for p, g in zip(params, step_grads):
                    p.data -= inner_lr * g
            if _it < self.mldg_inner_iters - 1:
                step_loss  = sum(self.quantile_loss(
                                 self.model(xe, xme, xde, xdme), ye) / n_sup
                             for xe, xme, xde, xdme, ye in support)
                step_grads = torch.autograd.grad(step_loss, params,
                                                 create_graph=False,
                                                 allow_unused=True)
                step_grads = [g if g is not None else torch.zeros_like(p)
                              for g, p in zip(step_grads, params)]

        # ── Query loss + gradients on updated params ──────────────────────
        qry_loss  = sum(self.quantile_loss(
                        self.model(xe, xme, xde, xdme), ye) / n_qry
                    for xe, xme, xde, xdme, ye in query)
        qry_grads = torch.autograd.grad(qry_loss, params,
                                        create_graph=False, allow_unused=True)
        qry_grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(qry_grads, params)]

        # ── H1: grad-alignment logging ────────────────────────────────────
        # Cos and norms of (∇F, β·∇G) tell us whether MLDG is doing anything
        # useful beyond inflating the gradient.  Gated by config flag.
        if self.mldg_log_grad_alignment:
            with torch.no_grad():
                sup_flat = torch.cat([g.detach().flatten() for g in sup_grads])
                qry_flat = torch.cat([g.detach().flatten() for g in qry_grads])
                sup_n = float(sup_flat.norm().item())
                qry_n = float(qry_flat.norm().item())
                dot   = float((sup_flat * qry_flat).sum().item())
                cos   = dot / (sup_n * qry_n + 1e-12)
                b     = float(self.mldg_trade_off)
                comb_n = float((sup_flat + b * qry_flat).norm().item())
            self._mldg_grad_log.append({
                "round":         self._mldg_current_round,
                "step":          len(self._mldg_grad_log) + 1,
                "cos":           cos,
                "sup_norm":      sup_n,
                "qry_norm":      qry_n,
                "combined_norm": comb_n,
                "n_sup_patients": int(n_sup),
                "n_qry_patients": int(n_qry),
            })

        # ── Restore params and apply combined gradient ────────────────────
        with torch.no_grad():
            for p, snap in zip(params, snapshots):
                p.data.copy_(snap)

        optimizer.zero_grad()
        for p, sg, qg in zip(params, sup_grads, qry_grads):
            p.grad = sg + self.mldg_trade_off * qg
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        return float(sup_loss.item() + self.mldg_trade_off * qry_loss.item())

    def _train_step_mldg(self, batch, optimizer) -> float:
        """
        Second-order MLDG via the `higher` library, with patient-disjoint
        support/query split.  Inner loop adapts a differentiable copy on
        support; outer loss flows gradients through that adaptation.
        """
        import higher

        x_enc, x_mark_enc, x_dec, x_mark_dec, y, _, _, pat_idx = \
            [t.to(self.device) for t in batch]

        split = self._split_by_patient(
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, pat_idx
        )
        if split is None:
            self._mldg_fallback_steps += 1
            return self._vanilla_step(
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, optimizer
            )
        self._mldg_used_steps += 1
        support, query = split
        n_sup, n_qry   = len(support), len(query)

        optimizer.zero_grad()

        # `higher` does not support AdamW; use SGD for the differentiable
        # inner loop (standard in MAML/MLDG). The outer optimizer is unchanged.
        inner_lr = self.mldg_inner_lr if self.mldg_inner_lr is not None else self.learning_rate
        inner_optimizer = optim.SGD(self.model.parameters(), lr=inner_lr)

        with higher.innerloop_ctx(self.model, inner_optimizer,
                                  copy_initial_weights=False) as (fmodel, diffopt):

            inner_lr_override = None
            for _ in range(self.mldg_inner_iters):
                inner_loss = sum(
                    self.quantile_loss(
                        fmodel(xe, xme, xde, xdme), ye
                    ) / n_sup
                    for xe, xme, xde, xdme, ye in support
                )
                diffopt.step(inner_loss, override=inner_lr_override)

            loss_outer = sum(
                self.quantile_loss(
                    self.model(xe, xme, xde, xdme), ye
                ) / n_sup
                for xe, xme, xde, xdme, ye in support
            )
            for xe, xme, xde, xdme, ye in query:
                loss_outer = loss_outer + self.mldg_trade_off * (
                    self.quantile_loss(fmodel(xe, xme, xde, xdme), ye) / n_qry
                )

        loss_outer.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return loss_outer.item()

    def _train_step_mldg_ref(self, batch, optimizer) -> float:
        """
        Reference-recipe second-order MLDG (matches the CGMTransformer-paper
        implementation).  Identical to `_train_step_mldg` EXCEPT the
        differentiable inner loop reuses the OUTER optimizer instead of a
        fresh plain SGD at `mldg_inner_lr`.  So meta-train and meta-test use
        the same update rule — the inner adaptation carries the outer
        optimizer's momentum / nesterov / weight-decay (SGD) or adaptive
        moments (Adam) and runs at the outer LR.  Everything else (split,
        outer-loss structure, grad clip) is left exactly as in
        `_train_step_mldg`.

        `higher` supports SGD and Adam (via DifferentiableSGD /
        DifferentiableAdam) but NOT AdamW.  This path therefore requires
        outer ∈ {SGD, Adam}; AdamW raises rather than silently misbehaving.
        """
        import higher

        x_enc, x_mark_enc, x_dec, x_mark_dec, y, _, _, pat_idx = \
            [t.to(self.device) for t in batch]

        split = self._split_by_patient(
            x_enc, x_mark_enc, x_dec, x_mark_dec, y, pat_idx
        )
        if split is None:
            self._mldg_fallback_steps += 1
            return self._vanilla_step(
                x_enc, x_mark_enc, x_dec, x_mark_dec, y, optimizer
            )
        self._mldg_used_steps += 1
        support, query = split
        n_sup, n_qry   = len(support), len(query)

        # `higher` supports SGD, Adam, Adamax, Adagrad, Adadelta, RMSprop, ...
        # but NOT AdamW (decoupled weight decay was never ported).  Block AdamW
        # explicitly; everything else flows through.
        if isinstance(optimizer, optim.AdamW):
            raise ValueError(
                "mldg_inner_reuse_outer_opt=True is incompatible with AdamW: "
                "`higher` does not implement a differentiable AdamW.  Use SGD "
                "or Adam — set proposer_train_optimizer_name='sgd' or 'adam'."
            )

        optimizer.zero_grad()

        # Inner loop reuses the OUTER optimizer via `higher`'s differentiable
        # wrapper — no fresh SGD, no separate mldg_inner_lr.
        with higher.innerloop_ctx(self.model, optimizer,
                                  copy_initial_weights=False) as (fmodel, diffopt):

            for _ in range(self.mldg_inner_iters):
                inner_loss = sum(
                    self.quantile_loss(
                        fmodel(xe, xme, xde, xdme), ye
                    ) / n_sup
                    for xe, xme, xde, xdme, ye in support
                )
                diffopt.step(inner_loss)

            loss_outer = sum(
                self.quantile_loss(
                    self.model(xe, xme, xde, xdme), ye
                ) / n_sup
                for xe, xme, xde, xdme, ye in support
            )
            for xe, xme, xde, xdme, ye in query:
                loss_outer = loss_outer + self.mldg_trade_off * (
                    self.quantile_loss(fmodel(xe, xme, xde, xdme), ye) / n_qry
                )

        loss_outer.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return loss_outer.item()

    # ------------------------------------------------------------------ helpers

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.optimizer_name.lower()
        params = self.model.parameters()
        if name == 'adam':
            return optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name == 'adamw':
            return optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif name == 'sgd':
            return optim.SGD(params, lr=self.learning_rate,
                             weight_decay=self.weight_decay, momentum=0.9)
        else:
            logger.warning(f"Unknown optimizer '{name}', using Adam")
            return optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def _build_scheduler(self, optimizer, round_idx: int = 0):
        stype = self.scheduler_type.lower()
        if stype == 'cosine':
            # Mirror FLockit_LoRA: single cosine span across ALL rounds × steps,
            # positioned at the current round's global step offset so LR is
            # continuous across FL rounds rather than resetting each round.
            steps_per_round = self.num_epochs * (self.max_steps_per_epoch or 500)
            total_steps     = (self.total_rounds + 1) * steps_per_round
            global_offset   = round_idx * steps_per_round
            lr_max          = self.learning_rate
            return optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda s: 0.5 * (1.0 + math.cos(math.pi * (global_offset + s) / total_steps)),
            )
        elif stype == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, self.num_epochs // 3), gamma=0.5)
        return None
