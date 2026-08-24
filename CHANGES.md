# CHANGES — data-quality corrections and revised results

## Code changes in this repository (branch `fix/data-quality-corrections`)

| file | change |
|---|---|
| `data_pipeline/build_metabonet.py` | timestamp conversion made unit-safe (`idx.as_unit("ns")`); previously `timestamp[us]` sources produced seconds / 1000 |
| `data_pipeline/build_clean_cohorts.py` | **new** - splices out constant-slope / sensor-clamped runs > 60 min (needed for HUPA-UCM, T1D-UOM); reuses manifest splits; packed output |
| `transformer/data_utils_transformer.py` | `y` now `[pred_len, 2]` = (target, `cgm_real` mask); time-of-day marks fall back to packed `tod_sin/tod_cos` channels and warn instead of silently using a synthetic phase |
| `transformer/flock_model_transformer.py` | `MSELossAdapter` / `QuantileLoss` compute a masked mean; validation `mse_norm` masked; test metrics unchanged |
| `data_pipeline/DATASET.md` (section 11), `README.md` | documentation of the three defects and the new build step |

The rest of this file is the results changelog handed to the paper revision.

---

For the session revising `glucose_fl_paper_working.tex`. Everything below comes from a
full rerun of the paper's experiment matrix (2026-08-21/22) after three data-quality
fixes. All numbers are 5-seed mean ± sd, RMSE in mg/dL, and were verified by
adversarial audit agents against the raw CSVs. Machine-readable source:
`output_clean_retrain/final_results_summary.csv`; narrative report: artifact
"The Clean Retrain".

---

## 1. What was wrong and what was fixed

Report these in the paper (methods/data section and/or a corrigendum note):

1. **HUPA-UCM arrived pre-fabricated from MetaboNet.** The public parquet has 0.0%
   missing CGM for all 22 patients: the 15-min FreeStyle Libre 2 data was linearly
   resampled to 5 min and dropout gaps were linearly bridged (runs up to 9 h)
   *upstream*, with no flag. 10.9% of samples lie in constant-slope runs; **39% of the
   original HUPA test windows contained invented data** (28.7% of train windows).
   No `cgm_real` mask could catch it (the mask is all-1 for this cohort).
   *Fix:* rebuilt HUPA-UCM (and T1D-UOM2) with the same pipeline plus a splice step:
   any constant-slope or sensor-clamped (≤40 / ≥400 mg/dL) run spanning **>60 min** is
   treated as missing; segments are cut there; clean stretches <280 samples dropped.
   Kept 88% of train / 70% of test samples. Same manifest patient splits.

2. **T1D-UOM's 15-minute patients were scored as real.** 7 of 14 patients are 15-min
   sampled; regridding to 5 min invents 2 of every 3 samples. These were correctly
   flagged (`cgm_real=0`, 28.5% of samples; **100% of the old T1D-UOM2 test windows had
   interpolated samples in the forecast target**) — but no training or evaluation code
   ever read the flag. *Fix:* training loss and validation model-selection are now
   masked to real samples (interpolated targets contribute zero gradient). Test-metric
   definitions are unchanged.

3. **ABC4D and ARISES timestamps were unix-seconds ÷ 1000.** Their staging parquets
   store `timestamp[us]`; the builder's `// 1_000_000_000` assumed nanoseconds. The
   time-of-day sin/cos marks were therefore near-constant garbage for these two
   cohorts **in every original run** (train and eval). *Fix:* timestamps reconstructed
   exactly (bound intersection + phase-snap to the intact tod channels; error 3e-8);
   builders patched to `idx.as_unit("ns")`.

Also: seeds are now **42–46** (the original set was 42/43/44/46/47 — 45 was skipped
by accident). Old references below remain over the original seed set.

**Confound note for the text:** the rerun changes three things at once (clean data,
masked loss, repaired marks). Old-vs-new deltas cannot be attributed to a single fix
without ablations (not run).

**Reporting standard:** all headline tables below are extended-metric evaluations on
the **clean test sets** (HUPA-UCM and T1D-UOM2 decontaminated; ABC4D/ARISES/OOD sets
unchanged — their contamination is <2% of windows at the 60-min threshold).
Evaluation logic is identical to the original pipeline (same code path,
recomputed pooled normalization; singles use own-cohort stats as before).

---

## 2. Replacement for Table `tab:main` (held-in, RMSE@30, 5-seed mean ± sd)

| strategy | HUPA-UCM | ABC4D | ARISES | T1D-UOM2 | average |
|---|---|---|---|---|---|
| Local (single-cohort) | 20.84 ±0.25 | 19.75 ±0.11 | **22.06 ±0.41** | 19.82 ±0.18 | 20.62 ±0.09 |
| FedAvg | 20.12 ±0.14 | 19.74 ±0.04 | 22.22 ±0.17 | 19.66 ±0.09 | 20.43 ±0.10 |
| FedProx (μ=0.05) | 20.36 ±0.29 | 19.87 ±0.18 | 22.35 ±0.28 | 19.76 ±0.21 | 20.58 ±0.23 |
| MLDG | **19.84 ±0.13** | **19.53 ±0.08** | **22.06 ±0.11** | 19.58 ±0.08 | **20.25 ±0.08** |
| APFL | 20.57 ±0.13 | 19.91 ±0.15 | 22.41 ±0.06 | 19.77 ±0.05 | 20.67 ±0.08 |
| APFL-decoupled | 20.28 ±0.15 | 19.82 ±0.12 | 22.24 ±0.18 | 19.68 ±0.12 | 20.51 ±0.12 |
| Ditto μ=0.01 | 20.15 ±0.25 | 19.69 ±0.08 | 22.15 ±0.27 | 19.61 ±0.17 | 20.40 ±0.19 |
| Ditto μ=0.1 | 20.19 ±0.23 | 19.68 ±0.08 | 22.14 ±0.23 | 19.60 ±0.14 | 20.40 ±0.16 |
| Ditto μ=1.0 | 20.24 ±0.08 | 19.72 ±0.08 | 22.23 ±0.12 | 19.66 ±0.09 | 20.46 ±0.08 |
| Centralized | 19.92 ±0.31 | 19.64 ±0.12 | 21.69 ±0.21 | **19.40 ±0.17** | 20.16 ±0.18 |

Notes for the text:
- MLDG's margin over FedAvg on the held-in average **doubled** (0.18 vs the old 0.09)
  with tighter seed spread — re-run the paired significance test (E5); it may now
  reach significance where the paper said it didn't.
- Local's ARISES win survives; local's HUPA number degrades most (its old value was
  flattered by training *and* testing on fabricated data).
- **Absolute HUPA-UCM values are ≈1.3 mg/dL higher across all arms** than published —
  a property of the honest test set (trivially predictable fabricated windows
  removed), not a modeling regression. Say this explicitly wherever old HUPA numbers
  were quoted (including `tab:prior` external comparisons — the GlucoFM benchmark
  evaluates on the fabricated-grid version of HUPA-UCM; consider a footnote).

## 3. Replacement for Table `tab:h60` (held-in, RMSE@60)

| strategy | HUPA-UCM | ABC4D | ARISES | T1D-UOM2 | average |
|---|---|---|---|---|---|
| Local | 36.76 ±0.39 | 33.86 ±0.19 | 36.26 ±0.55 | 32.28 ±0.13 | 34.79 ±0.03 |
| FedAvg | 35.48 ±0.17 | 33.69 ±0.10 | 36.19 ±0.15 | 32.25 ±0.08 | 34.40 ±0.09 |
| FedProx | 35.62 ±0.25 | 33.70 ±0.15 | 36.32 ±0.18 | 32.27 ±0.12 | 34.48 ±0.15 |
| MLDG | **35.11 ±0.11** | 33.37 ±0.12 | 36.11 ±0.10 | 32.19 ±0.08 | **34.19 ±0.07** |
| APFL | 35.77 ±0.17 | 34.04 ±0.31 | 36.07 ±0.10 | 32.20 ±0.08 | 34.52 ±0.11 |
| APFL-decoupled | 35.59 ±0.14 | 33.89 ±0.18 | 36.12 ±0.07 | 32.25 ±0.05 | 34.46 ±0.07 |
| Ditto μ=0.01 | 35.54 ±0.24 | 33.56 ±0.10 | 36.14 ±0.16 | 32.16 ±0.07 | 34.35 ±0.13 |
| Ditto μ=0.1 | 35.56 ±0.22 | 33.54 ±0.07 | 36.14 ±0.15 | 32.17 ±0.08 | 34.35 ±0.12 |
| Ditto μ=1.0 | 35.58 ±0.10 | 33.62 ±0.07 | 36.21 ±0.12 | 32.22 ±0.04 | 34.41 ±0.04 |
| Centralized | 35.58 ±0.26 | **33.36 ±0.15** | **35.85 ±0.15** | **31.98 ±0.10** | **34.19 ±0.11** |

This closes two gaps the old paper had: full per-cohort @60 for all strategies, and a
centralized row at 60 min (it exactly ties MLDG on the average).

## 4. Replacement for Table `tab:ood` (zero-shot OOD)

RMSE@30 (RMSE@60 in parentheses):

| model | ReplaceBG | BrisT1D | Flair |
|---|---|---|---|
| FedAvg | 21.48 ±0.12 (36.72 ±0.14) | 26.48 ±0.16 (44.63 ±0.18) | 25.18 ±0.14 (41.01 ±0.16) |
| FedProx | 21.65 ±0.21 (36.83 ±0.21) | 26.64 ±0.28 (44.70 ±0.28) | 25.32 ±0.22 (41.12 ±0.24) |
| MLDG | **21.30 ±0.07** (**36.38 ±0.03**) | 26.25 ±0.12 (44.17 ±0.06) | 25.02 ±0.09 (40.71 ±0.06) |
| Centralized | 21.41 ±0.23 (36.67 ±0.20) | 26.31 ±0.25 (44.32 ±0.24) | 25.00 ±0.23 (40.78 ±0.19) |
| single HUPA-UCM | 22.58 ±0.21 (38.48 ±0.28) | 27.85 ±0.20 (46.71 ±0.32) | 26.23 ±0.21 (42.82 ±0.30) |
| single ABC4D | 21.47 ±0.13 (36.62 ±0.12) | 26.42 ±0.16 (44.47 ±0.24) | 25.16 ±0.14 (40.92 ±0.19) |
| single ARISES | 21.71 ±0.19 (36.69 ±0.18) | **26.16 ±0.17** (**43.42 ±0.28**) | **24.84 ±0.09** (**40.06 ±0.15**) |
| single T1D-UOM2 | 21.83 ±0.13 (37.06 ±0.14) | 26.56 ±0.14 (44.25 ±0.24) | 25.11 ±0.13 (40.59 ±0.21) |

**REQUIRED narrative change — the ARISES-collapse claim must be retired.** The old
headline (single-ARISES OOD 28.51 ±6.85 / 32.39 ±5.84 / 30.99 ±6.23, seed-sd 30–40×
any federated model) was an artifact of the timestamp bug: that model trained on
degenerate time-of-day marks and met real marks OOD. With marks fixed, single-ARISES
is stable (sd ≈0.1–0.2) and is the **best single-cohort model OOD** — it even edges
MLDG on BrisT1D and Flair points. The corrected OOD framing:
- Federated models still beat the *average* single-cohort model on every OOD set,
  and MLDG is the best or tied-best global model throughout.
- The risk federation removes is **cohort selection**: you don't know in advance
  which single cohort transfers well, and the worst (single-HUPA, part-fabricated
  training data) is consistently ~1 mg/dL behind. Frame FL as insurance against the
  bad pick, not as "all locals collapse".
- Centralized transfers essentially as well as FedAvg but not better than MLDG —
  the old "centralized transfers worse than from-scratch on all three" (E9) should be
  softened to "centralized does not close the gap to target-trained models and offers
  no OOD advantage over MLDG".

## 5. Replacement for Table `tab:finetune` (E8, RMSE@30)

| target | from scratch | FedAvg→ft | FedProx→ft | MLDG→ft |
|---|---|---|---|---|
| ReplaceBG | 21.45 ±0.12 | 21.10 ±0.18 | **20.96 ±0.10** | 21.02 ±0.08 |
| BrisT1D | 26.03 ±0.12 | 25.75 ±0.05 | 25.80 ±0.05 | **25.74 ±0.06** |
| Flair | 24.52 ±0.06 | 24.37 ±0.14 | 24.32 ±0.11 | **24.23 ±0.06** |

Old refs (original seeds): scratch 21.38/26.25/24.50; best-ft 20.98/25.57/24.21.
- The conclusion "FL-pretrain→finetune beats from-scratch on every target" **holds**,
  and the best-method-per-target pattern is unchanged (FedProx on ReplaceBG, MLDG on
  BrisT1D and Flair).
- The BrisT1D margin shrank from 0.68 to 0.29 mg/dL: the masked loss repaired the
  from-scratch baseline (26.25 → 26.03; BrisT1D train targets were ~31% interpolated).
  Update any sentence quoting the old 0.68/0.7 margin.

## 6. Replacement for Fig `fig:dataeff` data (E10, MLDG-pretrain finetune, RMSE@30)

| target | scratch (100%) | 10% | 20% | 30% | 50% | 70% |
|---|---|---|---|---|---|---|
| ReplaceBG | 21.45 ±0.12 | 21.06 ±0.06 | 21.09 ±0.15 | 21.02 ±0.04 | 20.99 ±0.07 | 21.00 ±0.07 |
| BrisT1D | 26.03 ±0.12 | 26.09 ±0.23 | 26.01 ±0.21 | 25.86 ±0.19 | 25.79 ±0.17 | 25.70 ±0.20 |
| Flair | 24.52 ±0.06 | 24.33 ±0.11 | 24.33 ±0.13 | 24.24 ±0.08 | 24.25 ±0.08 | 24.24 ±0.06 |

**REQUIRED headline change.** The old headline ("on every OOD target, 10% of
patients + FL pretraining beats 100% from-scratch — e.g. 2 BrisT1D patients beat
15-patient local training") must be scoped:
- **ReplaceBG and Flair: fully holds.** Flat curves; 10% beats from-scratch at every
  fraction.
- **BrisT1D: 10% is now a statistical tie** (26.09 ±0.23 vs 26.03 ±0.12; per-seed
  deltas +0.24/−0.10/−0.24/−0.10/+0.52). The reliable win begins around 30% of
  patients (≈4–5 patients). The old 0.8-mg/dL win at 10% came from the
  contamination-weakened baseline.
- Suggested corrected headline: "on the two larger targets, ~10% of patients plus an
  FL-pretrained start matches or beats full local training; on the smallest cohort
  (15 patients) pretraining matches — but no longer substitutes for — most of the
  local data."

## 7. Diagnostics and secondary claims (all replicate)

- APFL's learned α still collapses toward the global model (0.02–0.08 by the final
  round in every seed).
- MLDG patient-disjoint meta-splits: 0 fallbacks to vanilla SGD (100% of batches).
- Best-model selection lands mid-schedule (e.g. seed 42: mldg/fedavg round 17 of 25).
- Ditto's μ ordering is flat within noise (all three μ within 0.06 of each other).

## 8. Things the paper text should add or adjust

1. A data-quality subsection (or corrigendum paragraph) describing the three bugs and
   fixes (§1) — the timestamp bug materially changed a published claim, so it should
   be disclosed, not silently fixed.
2. Seeds paragraph: 42–46 (note the original submission's 42/43/44/46/47).
3. Evaluation paragraph: clean-test extended metrics are the reporting standard;
   masked loss for training/validation, unmasked standard metrics at test.
4. `tab:prior` / `tab:oodprior` external comparisons: keep, but footnote that our
   corrected HUPA-UCM test set is harder than the benchmark's (fabricated windows
   removed), so cross-paper HUPA comparisons shifted ~1.3 mg/dL.
5. PFL rows everywhere use per-client personal models on their own cohorts (original
   protocol). Do not use averaged-personal-weights OOD numbers for PFL — a different
   quantity, excluded from these tables.
6. E5 significance test: recompute with the new per-seed values
   (`final_results_summary.csv` has the means; per-seed CSVs are in
   `output_clean_retrain/`).

## 9. Where every number comes from

- Sweep + CL per-seed CSVs: `output_clean_retrain/{cgm,pod_results}/...`
- Clean-test extended metrics (reporting): `output_clean_retrain/pod_results/clean_eval/seed_<S>/<arm>/extended_metrics.csv`
- Original-test evaluations of the new models (continuity check): `.../contaminated_eval/...`
- E8/E10: `output_clean_retrain/pod_results/followup/<job>/seed_<S>/best_model_local_test_irt.csv`
- Aggregates: `output_clean_retrain/final_results_summary.csv`
- Audit trail and orchestration: `ORCHESTRATION.md`; visual demo: `clean_demo/`
