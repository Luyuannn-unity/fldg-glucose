"""
Data utilities for CGMTransformer-based federated blood-glucose prediction.

Each FL client owns a `metabonet_splits/<source>/` directory containing:
  - segments_packed.npy         (T, F)     packed CGM (channel 0) + features
  - segments_offsets.npy        (n_seg+1,) cumulative segment lengths
  - segments_ts_packed.npy      (T,)       UNIX-second timestamps (optional)
  - window_index_<split>.npy    (N, 2|3)   [seg_idx, start_pos[, pat_idx]]
  - seg_split_map.json          {"train": [...], "val": [...], "test": [...]}

Windowing is stride-1 (driven entirely by the precomputed window_index files,
which match what FLockit_LoRA reads). Normalisation is z-score with
training-split statistics; values are denormalised at metric time using
mean/std carried per sample.
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from loguru import logger
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CGMTransformerDataConfig:
    seq_len:          int = 256
    label_len:        int = 6
    pred_len:         int = 24
    stride:           int = 1
    max_gap_steps:    int = 6        # retained for callers; honoured by preprocessing
    sampling_minutes: int = 5
    use_bolus:        bool = False   # if True, append log1p(bolus) as encoder channel 1


# ---------------------------------------------------------------------------
# Normalisation stats from a metabonet split
# ---------------------------------------------------------------------------

def compute_stats_from_metabonet_split(
    source_dir: str,
    split:      str = "train",
) -> Tuple[float, float]:
    """
    CGM mean and std from segments assigned to `split`. Used for per-client
    normalisation when no federation-wide global stats are supplied.
    """
    src         = Path(source_dir)
    split_map   = json.loads((src / "seg_split_map.json").read_text())
    seg_indices = split_map.get(split, [])
    if not seg_indices:
        logger.warning(f"No segments for split '{split}' in {source_dir} — using fallback stats")
        return 120.0, 50.0

    packed  = np.load(src / "segments_packed.npy", mmap_mode="r")
    offsets = np.load(src / "segments_offsets.npy")

    count, total, total_sq = 0, 0.0, 0.0
    for gi in seg_indices:
        s = int(offsets[gi])
        e = int(offsets[gi + 1])
        vals = packed[s:e, 0].astype(np.float64)
        count    += int(vals.shape[0])
        total    += float(vals.sum())
        total_sq += float((vals * vals).sum())

    if count == 0:
        return 120.0, 50.0
    mean = total / count
    var  = max(total_sq / count - mean * mean, 0.0)
    std  = max(float(np.sqrt(var)), 1e-6)
    logger.info(f"metabonet_splits/{src.name} [{split}] stats: mean={mean:.1f}, std={std:.1f}")
    return float(mean), std


# ---------------------------------------------------------------------------
# Lazy mmap-based dataset (mirrors LoRA's MetaboNetDataset / LoRAWindowDataset)
# ---------------------------------------------------------------------------

class LazyCGMTransformerDataset(Dataset):
    """
    Stride-1 sliding-window dataset for CGMTransformer.

    Reads `window_index_<split>.npy` ([seg_idx, start_pos[, pat_idx]]) and
    slices `segments_packed.npy` / `segments_ts_packed.npy` via mmap inside
    `__getitem__`. CGM is normalised and time-of-day sin/cos marks are
    computed on the fly; nothing is materialised up front.

    Returns the 8-tuple expected by FLockModelTransformer:
        x_enc, x_mark_enc, x_dec, x_mark_dec, y, cgm_mean, cgm_std, pat_idx
    """

    SECONDS_PER_DAY = 24 * 60 * 60
    MINUTES_PER_DAY = 24 * 60
    CGM_IDX         = 0     # CGM is channel 0 of segments_packed.npy
    BOLUS_IDX       = 3     # bolus IU at channel 3 (units, not normalised)
    CGM_REAL_IDX    = 6     # 1.0 = original sensor reading, 0.0 = interpolated
    TOD_SIN_IDX, TOD_COS_IDX = 1, 2   # precomputed time-of-day marks

    def __init__(
        self,
        source_dir: str,
        split:      str,
        config:     CGMTransformerDataConfig,
        cgm_mean:   float,
        cgm_std:    float,
    ):
        src = Path(source_dir)
        self._cfg        = config
        self._mean       = float(cgm_mean)
        self._std        = max(float(cgm_std), 1e-6)
        self._window_len = config.seq_len + config.pred_len

        idx_path = src / f"window_index_{split}.npy"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Missing {idx_path.name} in {src}. "
                "Run metabonet preprocessing to produce window indices."
            )
        self.index = np.load(idx_path)
        # Column 2 (patient index) is what MLDG's patient-disjoint support/query
        # split is built from. A 2-column index is accepted for FedAvg-family
        # methods, but silently makes every window look like one patient, which
        # would degrade MLDG into a vanilla step without any error. Expose the
        # fact so the model can refuse to run MLDG on such an index.
        self.has_patient_column = bool(self.index.ndim == 2 and self.index.shape[1] >= 3)

        packed_path  = src / "segments_packed.npy"
        offsets_path = src / "segments_offsets.npy"
        if not (packed_path.exists() and offsets_path.exists()):
            raise FileNotFoundError(
                f"Missing packed segments in {src}. "
                "Run pack_metabonet_segments.py with --overwrite."
            )
        self._packed  = np.load(packed_path,  mmap_mode="r")
        self._offsets = np.load(offsets_path)

        ts_path  = src / "segments_ts_packed.npy"
        self._ts = np.load(ts_path, mmap_mode="r") if ts_path.exists() else None

        self._split      = split
        self._source_dir = str(src)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        cfg       = self._cfg
        row       = self.index[i]
        seg_idx   = int(row[0])
        start     = int(row[1])
        pat_idx   = int(row[2]) if row.shape[0] >= 3 else -1
        seg_start = int(self._offsets[seg_idx])
        abs_start = seg_start + start
        end       = abs_start + self._window_len

        cgm = np.asarray(
            self._packed[abs_start:end, self.CGM_IDX], dtype=np.float32
        )
        cgm_norm = (cgm - self._mean) / self._std

        enc_cgm  = cgm_norm[:cfg.seq_len]
        fut_norm = cgm_norm[cfg.seq_len:cfg.seq_len + cfg.pred_len]

        # Encoder input: CGM only by default; with use_bolus, stack log1p(bolus)
        # over the *context* window as channel 1. Decoder stays CGM-only — bolus
        # is unknown over the prediction horizon at inference time.
        if cfg.use_bolus:
            bolus_ctx = np.asarray(
                self._packed[abs_start:abs_start + cfg.seq_len, self.BOLUS_IDX],
                dtype=np.float32,
            )
            bolus_ctx = np.log1p(np.clip(bolus_ctx, 0.0, None))
            enc_input = np.stack([enc_cgm, bolus_ctx], axis=-1)   # [seq_len, 2]
        else:
            enc_input = enc_cgm[:, None]                          # [seq_len, 1]

        # Informer-style decoder warm-up: last label_len of context, then zeros.
        dec_input = np.concatenate([
            enc_cgm[-cfg.label_len:],
            np.zeros(cfg.pred_len, dtype=np.float32),
        ])

        marks     = self._marks(abs_start, end)               # [window_len, 2]
        enc_marks = marks[:cfg.seq_len]
        dec_marks = np.concatenate([
            enc_marks[-cfg.label_len:],
            marks[cfg.seq_len:],
        ], axis=0)

        # y carries the cgm_real mask as channel 1 (1 = real sensor sample,
        # 0 = interpolated). The loss adapters use it for a masked mean so
        # interpolated targets never contribute gradient; every other consumer
        # indexes y[..., 0] explicitly.
        if self._packed.shape[1] > self.CGM_REAL_IDX:
            y_msk = np.asarray(
                self._packed[abs_start + cfg.seq_len:end, self.CGM_REAL_IDX],
                dtype=np.float32,
            )
        else:
            y_msk = np.ones(cfg.pred_len, dtype=np.float32)
        y_out = np.stack([fut_norm, y_msk], axis=-1)   # [pred_len, 2]

        return (
            torch.from_numpy(enc_input.astype(np.float32, copy=False)),
            torch.from_numpy(enc_marks.astype(np.float32, copy=False)),
            torch.from_numpy(dec_input[:, None].astype(np.float32, copy=False)),
            torch.from_numpy(dec_marks.astype(np.float32, copy=False)),
            torch.from_numpy(y_out.astype(np.float32, copy=False)),
            torch.tensor(self._mean, dtype=torch.float32),
            torch.tensor(self._std,  dtype=torch.float32),
            torch.tensor(pat_idx,    dtype=torch.int64),
        )

    def _marks(self, abs_start: int, end: int) -> np.ndarray:
        # Preferred: real unix timestamps. Fallback: the precomputed tod_sin/
        # tod_cos channels (always correct, unit-independent). Last resort:
        # a synthetic phase — which is WRONG (every window looks like it starts
        # at midnight), so it is logged loudly.
        if self._ts is not None:
            secs = np.asarray(self._ts[abs_start:end], dtype=np.int64) % self.SECONDS_PER_DAY
            mins = secs // 60
            frac = mins.astype(np.float32) / float(self.MINUTES_PER_DAY)
        elif self._packed.shape[1] > self.TOD_COS_IDX:
            return np.array(   # copy out of the read-only mmap
                self._packed[abs_start:end, self.TOD_SIN_IDX:self.TOD_COS_IDX + 1],
                dtype=np.float32,
            )
        else:
            if not getattr(self, "_warned_marks", False):
                logger.warning(f"{self._source_dir}: no timestamps and no tod channels — "
                               "using synthetic time-of-day marks (degenerate)")
                self._warned_marks = True
            n             = end - abs_start
            steps_per_day = self.MINUTES_PER_DAY / self._cfg.sampling_minutes
            frac          = (np.arange(n, dtype=np.float32) % steps_per_day) / steps_per_day
        return np.stack(
            [np.sin(2.0 * np.pi * frac), np.cos(2.0 * np.pi * frac)], axis=-1
        )

    def __repr__(self) -> str:
        return (
            f"LazyCGMTransformerDataset(source={self._source_dir!r}, split={self._split!r}, "
            f"n_windows={len(self):,}, mean={self._mean:.2f}, std={self._std:.2f})"
        )


def create_lazy_loaders_from_metabonet(
    source_dir:       str,
    config:           CGMTransformerDataConfig,
    cgm_mean:         float,
    cgm_std:          float,
    train_batch_size: int = 256,
    val_batch_size:   int = 256,
    test_batch_size:  int = 256,
    num_workers:      int = 0,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Build train/val/test DataLoaders backed by `LazyCGMTransformerDataset`.
    Returns None for any split whose window_index file is absent.
    """
    def _loader(split: str, batch_size: int, shuffle: bool):
        try:
            ds = LazyCGMTransformerDataset(source_dir, split, config, cgm_mean, cgm_std)
        except FileNotFoundError as e:
            logger.warning(str(e))
            return None
        if len(ds) == 0:
            return None
        return DataLoader(
            ds,
            batch_size=min(batch_size, len(ds)),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader("train", train_batch_size, shuffle=True),
        _loader("val",   val_batch_size,   shuffle=False),
        _loader("test",  test_batch_size,  shuffle=False),
    )
