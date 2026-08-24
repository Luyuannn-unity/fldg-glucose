# Federated CGM Forecasting with a Compact Transformer

Reproducibility code for **"Federated and meta-learned blood-glucose forecasting across sites: a domain-generalisation benchmark and a four-institution deployment"** (under review at PLOS Digital Health).

We train a single compact transformer (~4.9 M parameters) on continuous glucose
monitoring (CGM) data across four cohorts, using synchronous federated
averaging with optional MLDG (Meta-Learning for Domain Generalisation) on the
inner step. Each cohort's raw CGM never leaves its host — only 24-MB model
weights and a handful of scalar metrics cross the wire per round.

---

## Data quality and artifact handling

Two properties of the public CGM sources shape the pipeline:

- **Pre-resampled upstream data.** Some MetaboNet cohorts are already on a
  gap-free 5-min grid (e.g. HUPA-UCM: 15-min FreeStyle Libre 2 readings
  linearly resampled and gap-bridged before publication), so the `cgm_real`
  flag cannot identify their interpolated samples. `data_pipeline/build_clean_cohorts.py`
  therefore splices out constant-slope and sensor-clamped (≤40 / ≥400 mg/dL)
  runs longer than 60 min after interpolation. The paper's HUPA-UCM and T1D-UOM
  numbers use these spliced cohorts (Step 1b).
- **Masked objective.** Interpolated CGM samples (`cgm_real = 0`) never
  contribute to the training loss or to validation model selection; test
  metrics use the standard unmasked definitions.

`segments_ts_packed.npy` holds unix seconds and drives the time-of-day marks;
the loader falls back to the packed `tod_sin/tod_cos` channels if it is absent.

## Repo layout

```
.
├── README.md                          this file
├── requirements.txt
├── run_experiment.sh                  local single-machine reproduction
│
├── fl_server.py                       coordinator (HTTP, synchronous FedAvg)
├── fl_client.py                       one FL participant per cohort
├── fl_common.py                       shared client/server helpers
│
├── flock_sdk/                         FlockModel base class the client wraps
├── transformer/                       our compact transformer package
│   ├── flock_model_transformer.py     FLockModelTransformer: train/eval/aggregate hooks
│   ├── pfl_transformer.py             APFL / APFL-decoupled / DITTO personal branches
│   ├── model_hub_transformer.py       thin factory: name -> nn.Module
│   ├── data_utils_transformer.py      mmap-backed windowed CGM dataset + loaders
│   └── model/
│       ├── transformer_model.py       per-timestep encoder (main model)
│       ├── transformer_patch_model.py non-overlapping patched variant
│       └── layers/                    Embed, Attention, Encoder-Decoder blocks
│
├── config_{method}_seed{seed}.yaml    6 methods x 3 seeds = 18 configs
│     methods: fedavg fedprox mldg apfl apfl_decoupled ditto
│     seeds:   42 43 44
│
└── data_pipeline/                     raw data -> per-cohort packed .npy
    ├── DATASET.md                     full spec (channels, splits, gap handling)
    ├── build_metabonet.py             all publicly-available cohorts (incl. HUPA-UCM, T1D-UOM)
    ├── pack_metabonet_segments.py     shards -> single packed .npy per cohort
    └── metabonet_dataset.py           reference PyTorch Dataset (packed or sharded)
```

---

## Requirements

- Python 3.10 or 3.11
- NVIDIA GPU with **≥ 24 GB VRAM** (MLDG's second-order graph peaks around 20 GB
  at batch = 256). CPU-only works but is orders of magnitude slower.
- NVIDIA driver ≥ 525 (for the CUDA 12.1 PyTorch wheel).
- ~30 GB free disk (torch wheel + packed data + checkpoints).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Step 1 — Prepare the data

Full details are in [`data_pipeline/DATASET.md`](data_pipeline/DATASET.md).
Short version:

1. Put the raw MetaBoNet parquets at `./data_input/metabonet/`
   (`metabonet_public_train.parquet`, `metabonet_public_test.parquet`).
2. Build per-cohort splits + segment files:
   ```bash
   cd data_pipeline
   python build_metabonet.py
   # -> ../data_output/metabonet_splits/{HUPA-UCM,T1D-UOM,...}/
   ```
3. Pack per-cohort shards into a single mmap-friendly `.npy` per cohort
   (avoids file-descriptor exhaustion at scale):
   ```bash
   python pack_metabonet_segments.py \
       --root_dir ../data_output/metabonet_splits \
       --sources HUPA-UCM T1D-UOM \
       --overwrite
   ```

All splits share a common feature spec — see `DATASET.md` §3 for the
7-channel layout and §7 for the sliding-window shape.

> **A note on the paper's four cohorts.** The paper reports results on
> HUPA-UCM, ABC4D, ARISES, and T1D-UOM. Of these, **HUPA-UCM and T1D-UOM are
> extracted from the public MetaBoNet corpus** by `build_metabonet.py` and are
> fully reproducible here. **ABC4D and ARISES are proprietary** (access
> controlled by their respective study PIs) and their preprocessing scripts
> are not included. To reproduce those two arms exactly, obtain the raw data
> from the study owners; alternatively, the codebase runs on any set of
> cohorts you provide as long as each cohort directory conforms to the packed
> `.npy` layout documented in `DATASET.md` §5b.

---

### Step 1b — Artifact splicing for HUPA-UCM / T1D-UOM

```bash
cd data_pipeline
python build_clean_cohorts.py --cohorts HUPA-UCM T1D-UOM     --out ../data_output/metabonet_splits_clean --max-run-min 60
```

Point the training configs' `source_dirs` at `metabonet_splits_clean/<cohort>`
for these two cohorts. Verify any cohort's timestamps with
`np.diff(np.load('segments_ts_packed.npy')[:1000])` — values must be exactly 300
within a segment.

## Step 2 — Run one federated trial (local single-machine)

The paper's headline number is the seed-averaged test RMSE at a 30-minute
horizon across the four cohorts. To reproduce **one (method, seed) arm** on one
machine:

```bash
./run_experiment.sh                              # method=mldg, seed=42 (headline)
METHOD=fedavg          ./run_experiment.sh       # FedAvg baseline
METHOD=fedprox         ./run_experiment.sh       # FedProx (mu=0.05)
METHOD=apfl            ./run_experiment.sh       # APFL
METHOD=apfl_decoupled  ./run_experiment.sh       # APFL, decoupled personal branch
METHOD=ditto           ./run_experiment.sh       # DITTO (prox mu=0.1)
METHOD=apfl SEED=43    ./run_experiment.sh       # any (method, seed) combination
```

Configs ship for all six methods × three seeds (42, 43, 44) — 18 files named
`config_{method}_seed{seed}.yaml`. Each invocation spins up one server and one
client per cohort on `localhost:8088`, runs up to 25 rounds with early-stop
patience = 5, then evaluates the best-round model on each cohort's held-out
test split. Wall-clock is ~5-6 h per arm on an RTX 4090 (bottlenecked by the
synchronous FedAvg barrier, not by any individual client).

**Running with only the public cohorts.** The paper's federation is four
cohorts, two of which (ABC4D, ARISES) are proprietary — see
[`data_pipeline/DATASET.md`](data_pipeline/DATASET.md) §1. Override the cohort
list to run with whatever you actually have; `--num-clients` is derived from it:

```bash
COHORTS="HUPA-UCM T1D-UOM" ./run_experiment.sh              # public-data-only run
COHORTS="HUPA-UCM T1D-UOM" METHOD=ditto ./run_experiment.sh
```

Outputs land in `output_{method}_s{seed}/`:

- `best_model_test_metrics.csv` — per-cohort test RMSE (30-min horizon), MSE,
  TIR actual/predicted, computed on **the model that arm actually deploys**
- `round_summary.csv` — one row per round (avg val MSE/RMSE, new-best flag)
- `round_client_metrics.csv` — per-round per-client train loss + val metrics
- `best_global_model.pt` — the server-side FedAvg `w`-path checkpoint from the
  best round. For `fedavg`/`fedprox`/`mldg` this *is* the evaluated model. For
  the three personalised arms it is **not**: those deploy a per-client
  personal/mixture model that never leaves the client, so the test metrics come
  from those local models and no single global checkpoint reproduces them.
- `best_global_model_meta.txt` — best round + its avg val MSE

### Methods

| `METHOD` | Global aggregation | Personal branch | Deployed / evaluated model |
|---|---|---|---|
| `fedavg`          | FedAvg | — | global |
| `fedprox`         | FedAvg + prox term in the client's inner loop (`fedprox_mu`) | — | global |
| `mldg`            | FedAvg over a second-order MLDG inner loop (`higher`) | — | global |
| `apfl`            | FedAvg on `w` | `v_i` stepped along ∇f(v̄), α adapted per round | mixture `v̄ = αv + (1−α)w` |
| `apfl_decoupled`  | FedAvg on `w` | `v_i` trained on its own loss with its own persistent Adam | mixture `v̄ = αv + (1−α)w` |
| `ditto`           | FedAvg on `w` | `v_i` trained with prox toward the fresh global (`ditto_prox_mu`) | personal `v_i` |

`fedavg`, `fedprox`, and `mldg` produce a single global model, so their
`best_model_test_metrics.csv` is computed from the server's best global
checkpoint. The three personalised methods (`apfl`, `apfl_decoupled`, `ditto`)
deploy a **per-client** model, so each client evaluates its own personal or
mixture model — the model weights never leave the client, and the server still
only ever sees the FedAvg `w`-path. See `transformer/pfl_transformer.py` for
the implementation and `fl_client.py` for how a client snapshots its deploy
model on the server's best round (via `GET /round_info`).

---

## Step 3 — Distributed run (optional)

`fl_server.py` and `fl_client.py` are unchanged from Step 2 — the only
difference is that clients run on different machines and point at the server
over HTTP (typically tunnelled through SSH for firewall reasons). On the
server box:

```bash
python fl_server.py --config config_mldg_seed42.yaml \
    --host 0.0.0.0 --port 8088 --num-clients 4
```

On each client box (one per cohort):

```bash
python fl_client.py --config config_mldg_seed42.yaml \
    --source ./data_output/metabonet_splits/HUPA-UCM \
    --server http://SERVER_IP:8088 \
    --client-id HUPA-UCM
```

Set `--num-clients` to the number of cohorts actually participating (it
overrides the `server.num_clients` value in the config, which defaults to 4).

The client's data never leaves its own machine. Only serialised model weights
(24 MB per round, two-way) and small JSON scalars are transmitted. All
configs (seed, hyperparameters, per-cohort data channels) must match across
the fleet.

For the personalised methods (`apfl`, `apfl_decoupled`, `ditto`) this is
strictly stronger: the personal model `v_i` and the mixing weight `α_i` are
held only in the client process and are never serialised to the server. The
server still performs plain FedAvg on the `w`-path and is unaware of which
method the fleet is running, apart from clients calling `GET /round_info` to
learn which round was the global best so they can snapshot their own deploy
model.

---

## Method knobs

Shared across every config:

| Key | Value used in paper |
|---|---|
| `common_args.random_seed` | 42, 43, 44 |
| `server.num_rounds` | 25 |
| `server.early_stop_patience` | 5 |
| `train_args.proposer_train_batch_size` | 256 |
| `train_args.proposer_max_steps_per_epoch` | 500 (inner steps per round) |
| `train_args.proposer_learning_rate` | 1e-4 (Adam) |
| `model_args.d_model` / `e_layers` | 128 / 6 |

Method-specific keys (already set correctly in each shipped config):

| Key | Applies to | Value |
|---|---|---|
| `train_args.mldg` | mldg | `true` (`false` for every other method) |
| `train_args.mldg_first_order` | mldg | `false` — second-order via `higher` |
| `train_args.mldg_inner_reuse_outer_opt` | mldg | `true` |
| `train_args.federated_optimizer_name` | fedprox | `fedprox` (else `fedavg`) |
| `train_args.fedprox_mu` | fedprox | `0.05` |
| `train_args.pfl_method` | apfl / apfl_decoupled / ditto | the method name (else `null`) |
| `train_args.apfl_alpha_init` | apfl, apfl_decoupled | `0.25` |
| `train_args.apfl_alpha_lr` | apfl, apfl_decoupled | `0.1` |
| `train_args.ditto_prox_mu` | ditto | `0.1` |

---

## Cite

```bibtex
@article{PLACEHOLDER,
  title   = {PLACEHOLDER TITLE},
  author  = {PLACEHOLDER AUTHORS},
  journal = {PLACEHOLDER VENUE},
  year    = {2026}
}
```
