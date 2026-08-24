"""
build_metabonet.py

Extracts ReplaceBG and DCLP3 from MetaboNet, does a clean 80/10/10 patient-based
split, segments on CGM gaps > 30 min, and writes per-segment .npy files + window
indices ready for stride-1 sliding window training.

Output layout:
  dataset_from_scratch/metabonet_splits/
    manifest.json                    patient-to-split assignments
    {Source}/
      segments/
                seg_XXXXXX.npy               float32 [T, 7]  cgm | tod_sin | tod_cos | bolus | basal | insulin | cgm_real
        seg_XXXXXX_ts.npy            int64   [T]     unix timestamps (seconds)
      seg_split_map.json             {split: [global_seg_idx, ...]}
    window_index_train.npy         int32   [N, 3]  (seg_idx, start_pos, patient_int_idx)
      window_index_val.npy
      window_index_test.npy
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
TRAIN_PATH = Path("../data_input/metabonet/metabonet_public_train.parquet")
TEST_PATH  = Path("../data_input/metabonet/metabonet_public_test.parquet")
OUT_DIR    = Path("../data_output/metabonet_splits")

SOURCES = {
    "IOBP2":        {"basal_fill": "ffill"},   # AID, ~12% null basal
    "Flair":        {"basal_fill": "ffill"},   # AID, ~1% null basal
    "PEDAP":        {"basal_fill": "ffill"},   # AID, ~47% null basal
    "AZT1D":        {"basal_fill": "ffill"},   # AID, ~36% null basal
    "CTR3":         {"basal_fill": "zero"},    # AID, 100% null basal — nothing to forward-fill
    "BrisT1D":      {"basal_fill": "zero"},    # mixed, 100% null basal
    "T1D-UOM":      {"basal_fill": "zero"},    # MDI, ~98% null — do not ffill sparse injections
    "HUPA-UCM":     {"basal_fill": "zero"},    # MDI/SAP, 0% null — fill strategy irrelevant
    "ShanghaiT1DM": {"basal_fill": "zero"},    # MDI, ~59% null — do not ffill sparse injections
}
SEED       = 42
GAP_THRESH = 6        # 6 × 5-min intervals = 30 min
CTX        = 256
HORIZON    = 24
MIN_LEN    = CTX + HORIZON  # 280 rows minimum to yield any window

LOAD_COLS  = ["id", "source_file", "date", "CGM", "basal", "bolus", "insulin"]
# output feature order (index 0 = cgm is assumed by the Dataset below)
FEATURES   = ["cgm", "tod_sin", "tod_cos", "bolus", "basal", "insulin", "cgm_real"]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_source(source: str) -> pd.DataFrame:
    """Pool rows for `source` from both train and test parquets."""
    tr = pd.read_parquet(TRAIN_PATH, columns=LOAD_COLS)
    te = pd.read_parquet(TEST_PATH,  columns=LOAD_COLS)
    combined = pd.concat(
        [tr[tr["source_file"] == source],
         te[te["source_file"] == source]],
        ignore_index=True,
    )
    # seen patients appear in both parquets — deduplicate on (id, date)
    combined = (combined
                .drop_duplicates(subset=["id", "date"])
                .sort_values(["id", "date"])
                .reset_index(drop=True))
    return combined


def patient_split(patients: list, seed: int):
    rng = np.random.default_rng(seed)
    pts = list(patients)
    rng.shuffle(pts)
    n       = len(pts)
    n_test  = round(n * 0.1)
    n_val   = round(n * 0.1)
    n_train = n - n_val - n_test
    return pts[:n_train], pts[n_train:n_train + n_val], pts[n_train + n_val:]


def process_patient(df_pat: pd.DataFrame,
                    basal_fill: str = "ffill") -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Returns a list of (feat, ts) tuples — one per continuous segment.
            feat : float32 [T, 7]   cgm | tod_sin | tod_cos | bolus | basal | insulin | cgm_real
      ts   : int64   [T]      unix seconds
    Segments shorter than MIN_LEN are discarded.
    """
    df = df_pat.sort_values("date").copy()

    # ── snap to 5-min grid and reindex ───────────────────────────────────
    df["date"] = df["date"].dt.round("5min")
    df = df.drop_duplicates(subset="date").set_index("date")
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="5min")
    df = df.reindex(full_idx)
    df.index.name = "date"

    cgm = df["CGM"].values
    n   = len(cgm)

    # ── find segment boundaries (CGM NaN runs > GAP_THRESH intervals) ────
    boundaries = [0]
    i = 0
    while i < n:
        if np.isnan(cgm[i]):
            j = i
            while j < n and np.isnan(cgm[j]):
                j += 1
            if (j - i) > GAP_THRESH:
                boundaries.append(j)
            i = j
        else:
            i += 1
    boundaries.append(n)

    segments = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        seg = df.iloc[s:e].copy()

        if seg["CGM"].isna().all():
            continue

        # ── mark original-vs-imputed CGM and interpolate ─
        seg["cgm_real"] = (~seg["CGM"].isna()).astype(np.float32)
        seg["CGM"] = seg["CGM"].interpolate(method="linear", limit_area="inside")
        seg = seg.dropna(subset=["CGM"])

        if len(seg) < MIN_LEN:
            continue

        # ── fill insulin NaN ─────────────────────────────────────────────
        if basal_fill == "ffill":
            seg["basal"] = seg["basal"].ffill().fillna(0.0)
        else:
            seg["basal"] = seg["basal"].fillna(0.0)
        seg["bolus"]   = seg["bolus"].fillna(0.0)
        seg["insulin"] = seg["bolus"] + seg["basal"]  # recompute, don't trust raw column

        # ── time-of-day features ─────────────────────────────────────────
        idx  = seg.index
        secs = idx.hour * 3600 + idx.minute * 60 + idx.second
        tod  = secs / 86400.0
        seg["cgm"]     = seg["CGM"]
        seg["tod_sin"] = np.sin(2 * np.pi * tod)
        seg["tod_cos"] = np.cos(2 * np.pi * tod)

        feat = seg[FEATURES].values.astype(np.float32)          # [T, 6]
        ts   = (idx.as_unit("ns").astype(np.int64) // 1_000_000_000).values   # unit-safe → seconds (parquet may be timestamp[us])

        segments.append((feat, ts))

    return segments


def build_window_index(seg_indices: list[int],
                       seg_lengths: dict[int, int],
                       seg_patient_idx: dict[int, int]) -> np.ndarray:
    """
    stride-1 index: every valid (seg_idx, start_pos, patient_int_idx) triple.
    Returns int32 [N_windows, 3].
    """
    rows = []
    for gi in seg_indices:
        T = seg_lengths[gi]
        n_windows = T - MIN_LEN + 1
        if n_windows <= 0:
            continue
        starts = np.arange(n_windows, dtype=np.int32)
        gi_col = np.full(n_windows, gi, dtype=np.int32)
        pat_col = np.full(n_windows, seg_patient_idx[gi], dtype=np.int32)
        rows.append(np.stack([gi_col, starts, pat_col], axis=1))
    if not rows:
        return np.empty((0, 3), dtype=np.int32)
    return np.concatenate(rows, axis=0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    manifest_path = OUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for source, cfg in SOURCES.items():
        seg_dir = OUT_DIR / source / "segments"
        print(f"\n{'=' * 60}")
        print(f"  {source}  (basal_fill={cfg['basal_fill']})")
        print(f"{'=' * 60}")

        seg_dir.mkdir(parents=True, exist_ok=True)

        print("  Loading parquets...")
        df_src = load_source(source)
        patients = sorted(df_src["id"].unique())
        print(f"  Unique patients: {len(patients)}")

        train_pts, val_pts, test_pts = patient_split(patients, SEED)
        print(f"  Split: {len(train_pts)} train / {len(val_pts)} val / {len(test_pts)} test")
        patient_to_idx = {pid: i for i, pid in enumerate(patients)}

        manifest[source] = {
            "train": train_pts,
            "val":   val_pts,
            "test":  test_pts,
        }

        global_seg_idx  = 0
        seg_lengths     = {}
        seg_patient_idx = {}
        split_seg_idxs  = {"train": [], "val": [], "test": []}

        for split, pts in [("train", train_pts), ("val", val_pts), ("test", test_pts)]:
            n_patients_with_segs = 0
            for pid in pts:
                df_pat = df_src[df_src["id"] == pid]
                segs   = process_patient(df_pat, cfg["basal_fill"])
                if segs:
                    n_patients_with_segs += 1
                for feat, ts in segs:
                    np.save(seg_dir / f"seg_{global_seg_idx:06d}.npy",     feat)
                    np.save(seg_dir / f"seg_{global_seg_idx:06d}_ts.npy",  ts)
                    seg_lengths[global_seg_idx] = len(feat)
                    seg_patient_idx[global_seg_idx] = int(patient_to_idx[pid])
                    split_seg_idxs[split].append(global_seg_idx)
                    global_seg_idx += 1

            n_segs    = len(split_seg_idxs[split])
            n_windows = sum(max(0, seg_lengths[gi] - MIN_LEN + 1)
                            for gi in split_seg_idxs[split])
            print(f"  {split:5s}: {len(pts)} patients ({n_patients_with_segs} with valid segs)"
                  f" → {n_segs} segments → {n_windows:,} windows")

        # ── window indices ───────────────────────────────────────────────────
        for split in ("train", "val", "test"):
            idx = build_window_index(split_seg_idxs[split], seg_lengths, seg_patient_idx)
            np.save(OUT_DIR / source / f"window_index_{split}.npy", idx)
            print(f"  window_index_{split}.npy  shape={idx.shape}")

        with open(OUT_DIR / source / "seg_split_map.json", "w") as f:
            json.dump({s: idxs for s, idxs in split_seg_idxs.items()}, f)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Output: {OUT_DIR}")
    print(f"Features [{len(FEATURES)}]: {FEATURES}")
    print(f"Window shape per sample: [{CTX + HORIZON}, {len(FEATURES)}]  →  x=[{CTX},{len(FEATURES)}]  y=[{HORIZON}] (CGM only)")


if __name__ == "__main__":
    main()
