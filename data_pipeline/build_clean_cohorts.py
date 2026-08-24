"""build_clean_cohorts.py — rebuild cohorts with artifact splicing.

Same pipeline as build_metabonet.py (5-min grid, <=30-min gaps interpolated and
flagged cgm_real=0, >30-min gaps split, segments <280 rows dropped, stride-1
window index), plus one extra step required for cohorts whose UPSTREAM source is
already imputed (HUPA-UCM in MetaboNet has 0% missing CGM: 15-min sensor data
was linearly resampled and gap-bridged before publication, so cgm_real cannot
flag it; T1D-UOM's 15-min patients chain interpolation into long linear runs):

    after interpolation, any constant-slope run or any at/out-of-bounds
    (<=40 / >=400 mg/dL sensor clamp) run spanning more than --max-run-min
    minutes is treated as missing. The segment is cut there, the bad samples are
    dropped, and clean stretches shorter than MIN_LEN are discarded.

Patient splits are reused verbatim from metabonet_splits/manifest.json so the
clean cohorts stay comparable with the originals. Output is the packed format
(segments_packed.npy, segments_offsets.npy, segments_ts_packed.npy,
segments_packed_meta.json, seg_split_map.json, window_index_{train,val,test}.npy).

Usage (after build_metabonet.py has produced manifest.json):
    python build_clean_cohorts.py --cohorts HUPA-UCM T1D-UOM \
        --out ../data_output/metabonet_splits_clean --max-run-min 60

Effect on HUPA-UCM: ~29% of train and ~39% of test windows contain a >30-min
constant-slope run in the raw source (13% / 25% at the 60-min threshold); the
splice keeps ~88% of train and ~70% of test rows.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import build_metabonet as bm

CGM_LOW, CGM_HIGH, TOL = 40.0, 400.0, 0.5
DD_ATOL = 0.01


def runs_true(mask):
    if not mask.any():
        return
    m = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
    for s, e in zip(m[::2], m[1::2]):
        yield int(s), int(e)


def bad_sample_mask(cgm: np.ndarray, max_intervals: int) -> np.ndarray:
    """True where a sample is inside a constant-slope or OOB run spanning more
    than `max_intervals` five-minute intervals."""
    bad = np.zeros(len(cgm), dtype=bool)
    dd = np.diff(cgm, n=2)
    for s, e in runs_true(np.isclose(dd, 0.0, atol=DD_ATOL)):
        k = e - s                       # k zero second-diffs -> k+2 collinear points
        if k + 1 > max_intervals:
            bad[s:s + k + 2] = True
    oob = (cgm <= CGM_LOW + TOL) | (cgm >= CGM_HIGH - TOL)
    for s, e in runs_true(oob):
        if e - s - 1 > max_intervals:
            bad[s:e] = True
    return bad


def splice(feat, ts, max_intervals):
    bad = bad_sample_mask(feat[:, 0].astype(np.float64), max_intervals)
    pieces = []
    for s, e in runs_true(~bad):
        if e - s >= bm.MIN_LEN:
            pieces.append((feat[s:e], ts[s:e]))
    return pieces, int(bad.sum())


def build(cohort, source, manifest, out_root, basal_fill, max_intervals):
    print(f"\n=== {cohort} (source {source}) ===")
    df = bm.load_source(source)
    split_pts = manifest[cohort]
    all_pts = sorted(df["id"].unique())
    patient_to_idx = {pid: i for i, pid in enumerate(all_pts)}
    out_dir = Path(out_root) / cohort
    out_dir.mkdir(parents=True, exist_ok=True)

    feats, tss, seg_pat = [], [], []
    seg_split = {"train": [], "val": [], "test": []}
    gi = 0
    for split in ("train", "val", "test"):
        raw = kept = 0
        for pid in split_pts[split]:
            for feat, ts in bm.process_patient(df[df["id"] == pid], basal_fill):
                raw += len(feat)
                pieces, _ = splice(feat, ts, max_intervals)
                for pf, pt in pieces:
                    feats.append(pf.astype(np.float32))
                    tss.append(pt.astype(np.int64))
                    seg_split[split].append(gi)
                    seg_pat.append(int(patient_to_idx[pid]))
                    kept += len(pf)
                    gi += 1
        print(f"  {split:5s}: {raw:>9,} samples -> {kept:>9,} kept "
              f"({100 * kept / max(raw, 1):.1f}%), {len(seg_split[split])} segments")

    lengths = np.array([len(f) for f in feats], dtype=np.int64)
    offsets = np.zeros(len(feats) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    packed = np.concatenate(feats, axis=0)
    np.save(out_dir / "segments_packed.npy", packed)
    np.save(out_dir / "segments_offsets.npy", offsets)
    np.save(out_dir / "segments_ts_packed.npy", np.concatenate(tss, axis=0))
    (out_dir / "segments_packed_meta.json").write_text(json.dumps({
        "n_segments": len(feats), "total_rows": int(offsets[-1]),
        "n_features": len(bm.FEATURES), "has_timestamps": True,
        "format": "packed_v1", "cleaned": True,
        "max_run_min": max_intervals * 5}, indent=2))
    (out_dir / "seg_split_map.json").write_text(json.dumps(seg_split))
    for split in ("train", "val", "test"):
        rows = []
        for g in seg_split[split]:
            n_win = int(lengths[g]) - bm.MIN_LEN + 1
            if n_win > 0:
                rows.append(np.stack([np.full(n_win, g, np.int32),
                                      np.arange(n_win, dtype=np.int32),
                                      np.full(n_win, seg_pat[g], np.int32)], axis=1))
        idx = np.concatenate(rows, 0) if rows else np.empty((0, 3), np.int32)
        np.save(out_dir / f"window_index_{split}.npy", idx)
        print(f"  window_index_{split}.npy: {len(idx):,} windows")

    residual = sum(int(bad_sample_mask(packed[int(offsets[g]):int(offsets[g + 1]), 0]
                                       .astype(np.float64), max_intervals).sum())
                   for g in range(len(feats)))
    print(f"  residual bad samples after splice: {residual}")
    return residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=["HUPA-UCM", "T1D-UOM"],
                    help="manifest keys to rebuild (source_file == key unless "
                         "given as key=source, e.g. T1D-UOM2=T1D-UOM)")
    ap.add_argument("--manifest", default=str(bm.OUT_DIR / "manifest.json"))
    ap.add_argument("--out", default=str(bm.OUT_DIR.parent / "metabonet_splits_clean"))
    ap.add_argument("--max-run-min", type=int, default=60,
                    help="runs spanning more than this many minutes are spliced out")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    max_intervals = args.max_run_min // 5
    total = 0
    for spec in args.cohorts:
        cohort, _, source = spec.partition("=")
        source = source or cohort
        basal_fill = bm.SOURCES.get(source, {}).get("basal_fill", "zero")
        total += build(cohort, source, manifest, args.out, basal_fill, max_intervals)
    print(f"\nDone -> {args.out}  (total residual bad samples: {total})")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
