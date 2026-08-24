# Dataset Construction — Federated CGM Forecasting

This document describes how the training dataset was built for the federated
CGM (continuous glucose monitoring) forecasting experiments. All build scripts
live in `data_pipeline/`.

---

## 1. Data Sources

All publicly-reproducible cohorts are extracted from the MetaBoNet consolidated
parquet by `build_metabonet.py`. Each cohort inherits its `basal_fill` strategy
from the source's insulin-delivery modality (see §3a).

These are exactly the entries of the `SOURCES` dict in `build_metabonet.py`:

| Source        | Insulin modality | Basal fill |
|---------------|------------------|------------|
| IOBP2         | AID              | ffill      |
| Flair         | AID              | ffill      |
| PEDAP         | AID              | ffill      |
| AZT1D         | AID              | ffill      |
| CTR3          | AID (100% null basal) | zero  |
| BrisT1D       | mixed (100% null basal) | zero |
| **HUPA-UCM**  | MDI / SAP        | zero       |
| **T1D-UOM**   | MDI              | zero       |
| ShanghaiT1DM  | MDI              | zero       |

- Raw data: `data_input/metabonet/metabonet_public_train.parquet` +
  `metabonet_public_test.parquet`
- MetaBoNet ships its own chronological train/test split with "seen" patients
  appearing in the test set. **We do not use that split.** We pool both
  parquets, deduplicate on `(id, date)`, and apply a clean patient-level
  split (see §2).
- The paper's headline cohorts **HUPA-UCM** and **T1D-UOM** are both built
  directly by `build_metabonet.py`.

> **A note on ABC4D and ARISES.** The paper additionally reports on two
> cohorts, **ABC4D** and **ARISES**, that are access-controlled by their
> respective study PIs and are therefore not publicly reproducible. Their
> preprocessing scripts are not included in this release. To reproduce those
> arms, obtain the raw data from the study owners and produce a
> `data_output/metabonet_splits/{ABC4D,ARISES}/` directory that matches the
> packed layout in §5b. Everything downstream of the pipeline is agnostic to
> the cohort provenance.

---

## 2. Patient-Level Splits

All splits are **patient-level** — a patient appears in exactly one of train /
val / test. This prevents data leakage and avoids the MetaBoNet "seen patient"
problem. The default split is 80/10/10 with `seed=42` on the pooled patient
list per source.

---

## 3. Feature Engineering

Each segment file is a `float32 [T, 7]` array with the following channels:

| Index | Name        | Description |
|-------|-------------|-------------|
| 0     | `cgm`       | CGM reading (mg/dL), interpolated within segments (see §4) |
| 1     | `tod_sin`   | `sin(2π × seconds_of_day / 86400)` — cyclic time-of-day |
| 2     | `tod_cos`   | `cos(2π × seconds_of_day / 86400)` |
| 3     | `bolus`     | Bolus insulin (IU per 5-min slot); NaN-filled to 0 |
| 4     | `basal`     | Basal insulin (IU per 5-min slot); fill strategy varies by source (see §3a) |
| 5     | `insulin`   | `bolus + basal` — always recomputed, never taken from raw column |
| **6** | **`cgm_real`** | **1.0 if CGM is an original sensor reading, 0.0 if linearly interpolated** |

> **Column 6 (`cgm_real`) was added to enable masked loss during training**: the forecast horizon (indices CTX … CTX+HORIZON) may contain interpolated CGM values. Loss should only be computed on steps where `cgm_real == 1.0`.

### 3a. Basal fill strategy

The correct fill strategy depends on the insulin delivery modality. The exact
per-source assignment is the `SOURCES` dict in `build_metabonet.py` (reproduced
in §1); the rule behind it is:

| Strategy | Applies to | Reason |
|----------|-----------|--------|
| forward-fill → 0 | pump / AID sources (IOBP2, Flair, PEDAP, AZT1D) | Basal rate is continuous and recorded per interval, so carrying the last rate forward is physiologically correct |
| fill with 0 | MDI sources (HUPA-UCM, T1D-UOM, ShanghaiT1DM) and sources with 100% null basal (CTR3, BrisT1D) | Long-acting basal is recorded as a single sparse injection event. Forward-filling e.g. an 82 IU spike across the following 24 hours would be physiologically wrong |

### 3b. Insulin recomputation

`insulin = bolus + basal` is **always recomputed** after filling NaN values. The raw `insulin` column in MetaboNet is not used because it can contain NaN where basal/bolus were NaN, leading to inconsistency.

---

## 4. Gap Handling and Segmentation

CGM traces have gaps (sensor dropouts, calibration periods). Strategy:

1. Snap all timestamps to the nearest 5-minute grid.
2. Reindex to a full uniform 5-min grid (filling gaps with NaN).
3. Identify NaN runs in the CGM column:
   - **Gap ≤ 30 min (≤ 6 intervals)**: linearly interpolate CGM; mark interpolated steps with `cgm_real = 0`.
   - **Gap > 30 min**: split the trace — start a new segment after the gap.
4. Drop segments shorter than `CTX + HORIZON = 280` rows (cannot yield any training window).
5. Drop segments where all CGM values are NaN.

This produces variable-length segment arrays stored as individual `.npy` files.

---

## 5. Output Format

### 5a. Per-segment files (original sharded format)

```
metabonet_splits/
  manifest.json                         patient-to-split assignments for all sources
  {Source}/
    segments/
      seg_000000.npy                    float32 [T, 7]  — feature array
      seg_000000_ts.npy                 int64   [T]     — unix timestamps (seconds)
      ...
    seg_split_map.json                  {split: [global_seg_idx, ...]}
    window_index_train.npy              int32   [N, 3]  — (seg_idx, start_pos, patient_int_idx)
    window_index_val.npy
    window_index_test.npy
```

> **The third column is mandatory for MLDG.** `patient_int_idx` is what MLDG's
> patient-disjoint support/query split is built from. A 2-column index is
> accepted by the dataset for the FedAvg-family methods, but with `mldg: true`
> the client raises rather than running, because every window would report the
> same patient id and each MLDG step would silently degrade to a vanilla step.

### 5b. Packed format (recommended — avoids too-many-open-files)

Running `pack_metabonet_segments.py` concatenates all per-segment files into two arrays per source:

```
{Source}/
  segments_packed.npy                   float32 [sum(T_i), 7]  — all segments concatenated
  segments_offsets.npy                  int64   [n_segments + 1]  — segment start offsets
  segments_ts_packed.npy                int64   [sum(T_i)]       — timestamps concatenated
  segments_packed_meta.json
```

`segments_offsets[i]` is the row index in `segments_packed` where segment `i` begins.  
`segments_offsets[i+1] - segments_offsets[i]` gives its length.

To access segment `i`, start position `start`, window length `W`:
```python
s = offsets[seg_idx]
window = packed[s + start : s + start + W]
```

**Why packed?** A large source can produce many thousands of segments, and the
sharded format needs one simultaneously-open mmap file descriptor per segment
per dataset instance. Running several FL clients in one process multiplies that
by (clients × splits) and quickly exceeds practical fd limits. The packed
format reduces this to 2 fds per instance regardless of segment count.

Run packing:
```bash
cd data_pipeline
python pack_metabonet_segments.py \
    --root_dir ../data_output/metabonet_splits \
    --sources HUPA-UCM T1D-UOM \
    --overwrite
```

---

## 6. Sliding Window Index

The window index (`window_index_{split}.npy`) is an `int32 [N, 3]` array where
each row is `(seg_idx, start_pos, patient_int_idx)`.

- **Stride = 1**: every valid start position in every segment is indexed. This maximises data utilisation.
- A window is valid when `start_pos + CTX + HORIZON ≤ len(segment)`.
- Windows from val patients are indexed in `window_index_val.npy`, etc. — the window index is the only thing that changes between splits; segment files are shared.
- `patient_int_idx` identifies which patient the window came from. It is
  **required** whenever `train_args.mldg` is true (see the note in §5a).

Storage cost: 12 bytes per window, versus roughly 6.7 KB if each window were
pre-materialised — about three orders of magnitude smaller.

---

## 7. Window Shape

| Tensor  | Shape          | Type    | Contents |
|---------|----------------|---------|----------|
| `x`     | `[256, 7]`     | float32 | Context window (all 7 channels including `cgm_real`) |
| `y`     | `[24]`         | float32 | CGM forecast horizon (channel 0) |
| `y_msk` | `[24]`         | float32 | 1 = original CGM, 0 = interpolated (channel 6 of the horizon rows) |

Models using the dataset should compute loss only on `y_msk == 1` entries.

> **Relationship to the FL configs.** The `CTX = 256` / `HORIZON = 24` constants
> above are the *build-time* window budget: `build_metabonet.py` drops segments
> shorter than `CTX + HORIZON = 280` rows and only indexes start positions with
> that much room. The FL configs train at `seq_len: 72` / `pred_len: 12`
> (window length 84), which is strictly smaller, so every indexed window is
> valid at training time. The build-time constants are deliberately
> conservative — they let the same prebuilt index serve longer-context
> experiments without a rebuild.

---

## 8. Window Counts (exact, seed=42)

Window counts for the two publicly-reproducible headline cohorts, produced by
`build_metabonet.py` with `seed=42`:

| Source    | Train windows | Val windows | Test windows | Total     |
|-----------|---------------|-------------|--------------|-----------|
| HUPA-UCM  | 244,630       | (see cohort dir) | (see cohort dir) | — |
| T1D-UOM   | 213,758       | (see cohort dir) | (see cohort dir) | — |

The remaining sources in `SOURCES` (IOBP2, Flair, PEDAP, AZT1D, CTR3, BrisT1D,
ShanghaiT1DM) are produced by the same build script and follow the same layout;
they are not used by the paper's four-cohort federation.

---

## 9. Build Scripts

| Script | Purpose |
|--------|---------|
| `build_metabonet.py`         | Extract every MetaBoNet source into per-segment `.npy` files with the seed-42 patient-level split |
| `pack_metabonet_segments.py` | Pack per-segment shards into a single `segments_packed.npy` per source (required for multi-client FL) |
| `metabonet_dataset.py`       | Reference `MetaboNetDataset` PyTorch Dataset (auto-detects packed vs sharded) |

### Running order

```bash
# 1. Build all MetaBoNet-derived cohorts
python build_metabonet.py

# 2. Pack the two publicly-reproducible cohorts used in the paper
python pack_metabonet_segments.py \
    --root_dir ../data_output/metabonet_splits \
    --sources HUPA-UCM T1D-UOM \
    --overwrite
```

---

## 10. Key Design Decisions and Gotchas

- **No MetaBoNet default split**: MetaBoNet's train/test parquets share patients ("seen patients" in test). We pool both and re-split by patient.
- **Deduplication**: `(id, date)` deduplication is applied after pooling train+test parquets, since some patients appear in both.
- **Basal fill by modality**: pump / AID sources use forward-fill; MDI sources (HUPA-UCM, T1D-UOM, ShanghaiT1DM) fill with 0 — long-acting basal is recorded as a sparse injection event, and forward-filling would incorrectly propagate the injection across the next many hours.
- **Insulin recomputation**: `insulin = bolus + basal` is always recomputed. The raw `insulin` column is unreliable after NaN filling.
- **`cgm_real` mask**: Column 6 is 1 for original sensor readings, 0 for linearly-interpolated values. Use this as a loss mask so the model is not penalised for errors on imputed targets.
- **Packed format required for multi-client FL**: The sharded format opens one fd per segment. Always pack before running FL runs with more than one client.

## 11. Artifact splicing and pipeline invariants

### 11a. Pre-resampled upstream sources
Some public sources are already on a gap-free 5-min grid. MetaboNet's HUPA-UCM,
for example, has no missing CGM at all: its 15-min FreeStyle Libre 2 readings were
linearly resampled and dropout gaps linearly bridged (runs up to several hours)
before publication. Such samples arrive as ordinary non-NaN values, so the §4 gap
logic never triggers and `cgm_real` stays 1 — roughly 29% of HUPA-UCM training
windows and 39% of its test windows contain a constant-slope run longer than
30 min. T1D-UOM's 15-min-sampled patients (7 of 14) are a second case: regridding
interpolates two of every three samples, and the bridges can chain into runs longer
than the 30-min gap rule.

**`build_clean_cohorts.py`** (run after `build_metabonet.py`) handles both: after
interpolation, any constant-slope run or any at/out-of-bounds (≤40 / ≥400 mg/dL
sensor clamp) run spanning more than 60 min is treated as missing; segments are
spliced there; sub-280-row remnants are dropped; patient splits are reused from
`manifest.json`. The paper's HUPA-UCM and T1D-UOM numbers use these spliced
cohorts (the other cohorts are <2% affected at this threshold).

### 11b. Timestamp units
`segments_ts_packed.npy` holds **unix seconds**. `build_metabonet.py` converts the
index with `idx.as_unit("ns")` so the result is independent of the parquet's
timestamp resolution (`timestamp[us]` sources are common). Sanity check for any
built cohort: `np.diff(ts)` within a segment must be exactly 300.

### 11c. The `cgm_real` mask in training
The dataset returns `y` with the mask as a second channel (`[pred_len, 2]`); the
loss adapters compute a masked mean and the validation `mse_norm` is masked, so
interpolated targets never contribute gradient or drive model selection.
Headline test metrics keep their standard unmasked definitions.
