"""
PyTorch Dataset for the metabonet_splits directory produced by build_metabonet.py.

Usage:
    from dataset_from_scratch.metabonet_dataset import MetaboNetDataset
    from torch.utils.data import DataLoader

    ds = MetaboNetDataset("dataset_from_scratch/metabonet_splits/ReplaceBG", split="train")
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    x, y, m, p = next(iter(dl))
    # x: [B, 256, F]  float32  — full context window (supports optional cgm_real feature)
    # y: [B, 24]      float32  — CGM forecast horizon
    # m: [B, 24]      float32  — 1=real CGM target, 0=imputed CGM target
    # p: [B]          int32    — patient integer index (or -1 if unavailable)
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class MetaboNetDataset(Dataset):
    """
    Stride-1 sliding window dataset over pre-segmented .npy files.

    Each __getitem__ is a zero-copy numpy mmap slice — no data is duplicated
    on disk regardless of stride.
    """

    CTX     = 256
    HORIZON = 24
    # feature index of CGM channel
    CGM_IDX = 0
    # optional feature index when builders include cgm_real column
    CGM_REAL_IDX = 6

    def __init__(self, source_dir: str | Path, split: str):
        src = Path(source_dir)

        # window index: int32 [N_windows, 2 or 3]
        #   [seg_idx, start_pos] or [seg_idx, start_pos, patient_int_idx]
        self.index = np.load(src / f"window_index_{split}.npy")

        # Preferred packed format: one contiguous segment array + offsets.
        self._packed = None
        self._offsets = None
        self.segs = None
        packed_path = src / "segments_packed.npy"
        offsets_path = src / "segments_offsets.npy"
        if packed_path.exists() and offsets_path.exists():
            self._packed = np.load(packed_path, mmap_mode="r")
            self._offsets = np.load(offsets_path)
        else:
            # Backward-compatible fallback: memory-map each segment file.
            seg_dir = src / "segments"
            seg_files = sorted(f for f in seg_dir.glob("seg_*.npy")
                               if "_ts" not in f.name)
            self.segs = [np.load(f, mmap_mode="r") for f in seg_files]

        self._window_len = self.CTX + self.HORIZON
        self._split      = split
        self._source_dir = str(src)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        row = self.index[i]
        seg_idx = int(row[0])
        start = int(row[1])
        pat_idx = int(row[2]) if row.shape[0] >= 3 else -1
        if self._packed is not None:
            seg_start = int(self._offsets[seg_idx])
            abs_start = seg_start + start
            window = self._packed[abs_start : abs_start + self._window_len]
        else:
            window = self.segs[seg_idx][start : start + self._window_len]  # mmap slice → view
        x = torch.from_numpy(window[: self.CTX].copy())                # [256, F]
        y = torch.from_numpy(window[self.CTX :, self.CGM_IDX].copy())  # [24]
        if window.shape[1] > self.CGM_REAL_IDX:
            y_mask = torch.from_numpy(window[self.CTX :, self.CGM_REAL_IDX].copy())
        else:
            y_mask = torch.ones(self.HORIZON, dtype=torch.float32)
        return x, y, y_mask, torch.tensor(pat_idx, dtype=torch.int32)

    def __repr__(self) -> str:
        if self._offsets is not None:
            n_segments = len(self._offsets) - 1
            storage = "packed"
        else:
            n_segments = len(self.segs)
            storage = "multi-file"
        return (f"MetaboNetDataset(source={self._source_dir!r}, split={self._split!r}, "
                f"n_windows={len(self):,}, n_segments={n_segments}, storage={storage})")
