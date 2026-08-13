"""
Pack metabonet_splits segment shards into a single memory-mappable array per source.

Input (existing format):
  {source_dir}/segments/seg_XXXXXX.npy
  {source_dir}/segments/seg_XXXXXX_ts.npy  (optional)

Output (packed format):
  {source_dir}/segments_packed.npy          float32 [sum(T_i), F]
  {source_dir}/segments_offsets.npy         int64   [n_segments + 1]
  {source_dir}/segments_ts_packed.npy       int64   [sum(T_i)]      (optional)
  {source_dir}/segments_packed_meta.json

This allows training code to keep only one data file open instead of tens of
thousands of segment files.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SEG_RE = re.compile(r"^seg_(\d{6})\.npy$")


def _collect_segment_files(seg_dir: Path) -> dict[int, Path]:
    seg_files: dict[int, Path] = {}
    for p in seg_dir.glob("seg_*.npy"):
        if p.name.endswith("_ts.npy"):
            continue
        m = SEG_RE.match(p.name)
        if m is None:
            continue
        idx = int(m.group(1))
        seg_files[idx] = p
    return seg_files


def _infer_expected_count(source_dir: Path, seg_files: dict[int, Path]) -> int:
    candidates = []
    for split in ("train", "val", "test"):
        p = source_dir / f"window_index_{split}.npy"
        if p.exists():
            idx = np.load(p, mmap_mode="r")
            if idx.size:
                candidates.append(int(idx[:, 0].max()) + 1)

    split_map = source_dir / "seg_split_map.json"
    if split_map.exists():
        data = json.loads(split_map.read_text())
        flat = []
        for split in ("train", "val", "test"):
            flat.extend(data.get(split, []))
        if flat:
            candidates.append(max(flat) + 1)

    if candidates:
        return max(candidates)
    if not seg_files:
        return 0
    return max(seg_files) + 1


def pack_source(source_dir: Path, include_ts: bool, overwrite: bool, remove_original: bool) -> None:
    seg_dir = source_dir / "segments"
    if not seg_dir.exists():
        raise FileNotFoundError(f"Missing segments directory: {seg_dir}")

    seg_files = _collect_segment_files(seg_dir)
    if not seg_files:
        raise RuntimeError(f"No segment files found in {seg_dir}")

    expected = _infer_expected_count(source_dir, seg_files)
    missing = [i for i in range(expected) if i not in seg_files]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} segment files in {source_dir}. "
            f"First missing index: {missing[0]}"
        )

    packed_path = source_dir / "segments_packed.npy"
    offsets_path = source_dir / "segments_offsets.npy"
    ts_packed_path = source_dir / "segments_ts_packed.npy"
    meta_path = source_dir / "segments_packed_meta.json"

    outputs = [packed_path, offsets_path, meta_path]
    if include_ts:
        outputs.append(ts_packed_path)

    if not overwrite:
        for out in outputs:
            if out.exists():
                raise FileExistsError(f"Output exists (use --overwrite): {out}")

    n_segments = expected
    lengths = np.zeros(n_segments, dtype=np.int64)
    n_features = None

    for i in range(n_segments):
        arr = np.load(seg_files[i])   # regular load — sequential, fd closed each iteration
        if arr.ndim != 2:
            raise ValueError(f"{seg_files[i]} must be 2D, got shape={arr.shape}")
        if n_features is None:
            n_features = int(arr.shape[1])
        elif int(arr.shape[1]) != n_features:
            raise ValueError(
                f"Feature mismatch at {seg_files[i]}: expected {n_features}, got {arr.shape[1]}"
            )
        lengths[i] = int(arr.shape[0])

    assert n_features is not None
    offsets = np.zeros(n_segments + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    total_rows = int(offsets[-1])

    # Allocate in RAM and write with np.save — avoids SIGBUS on NFS when using
    # open_memmap(mode="w+"), which requires the network filesystem to back mmap writes.
    packed = np.empty((total_rows, n_features), dtype=np.float32)

    ts_packed = np.empty(total_rows, dtype=np.int64) if include_ts else None

    for i in range(n_segments):
        s, e = int(offsets[i]), int(offsets[i + 1])
        packed[s:e] = np.load(seg_files[i]).astype(np.float32, copy=False)
        if ts_packed is not None:
            ts_file = seg_dir / f"seg_{i:06d}_ts.npy"
            if ts_file.exists():
                ts = np.load(ts_file)
                if len(ts) != (e - s):
                    raise ValueError(f"Timestamp length mismatch for {ts_file}")
                ts_packed[s:e] = ts.astype(np.int64, copy=False)
            else:
                ts_packed[s:e] = -1

    np.save(packed_path, packed)
    del packed
    if ts_packed is not None:
        np.save(ts_packed_path, ts_packed)
        del ts_packed

    np.save(offsets_path, offsets)

    meta = {
        "n_segments": n_segments,
        "total_rows": total_rows,
        "n_features": n_features,
        "has_timestamps": bool(include_ts),
        "format": "packed_v1",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    if remove_original:
        for i in range(n_segments):
            seg_files[i].unlink(missing_ok=False)
            ts_file = seg_dir / f"seg_{i:06d}_ts.npy"
            if ts_file.exists():
                ts_file.unlink()

    print(
        f"[OK] {source_dir.name}: segments={n_segments:,}, rows={total_rows:,}, "
        f"features={n_features}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path("../data_output/metabonet_splits"),
        help="Root containing per-source folders",
    )
    parser.add_argument(
        "--sources",
        type=str,
        nargs="*",
        default=None,
        help="Specific source folder names. If omitted, pack all folders under root_dir.",
    )
    parser.add_argument("--no_ts", action="store_true", help="Do not pack *_ts.npy files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing packed files")
    parser.add_argument(
        "--remove_original",
        action="store_true",
        help="Delete original seg_*.npy and seg_*_ts.npy after successful packing",
    )
    args = parser.parse_args()

    if args.sources:
        source_dirs = [args.root_dir / s for s in args.sources]
    else:
        source_dirs = [p for p in sorted(args.root_dir.iterdir()) if p.is_dir() and (p / "segments").exists()]

    if not source_dirs:
        raise RuntimeError(f"No source directories found under {args.root_dir}")

    for src in source_dirs:
        pack_source(
            source_dir=src,
            include_ts=not args.no_ts,
            overwrite=args.overwrite,
            remove_original=args.remove_original,
        )

    print("Done.")


if __name__ == "__main__":
    main()
