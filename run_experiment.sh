#!/usr/bin/env bash
# Local reproduction: run one 4-cohort federated trial on ONE machine.
#
# Server + 4 clients (HUPA-UCM, ABC4D, ARISES, T1D-UOM) all talk to localhost.
# Same code path as the multi-machine version — a real distributed run only
# changes the --server URL each client points at.
#
# Usage:
#   ./run_experiment.sh                                       # method=mldg, seed=42
#   METHOD=fedavg ./run_experiment.sh                         # {fedavg,fedprox,mldg,apfl,apfl_decoupled,ditto}
#   METHOD=apfl SEED=43 ./run_experiment.sh                   # any (method, seed) combo we ship a config for
#   PORT=9001 METHOD=ditto SEED=42 ./run_experiment.sh        # override port too
#
# COHORTS: the paper's federation is the 4 cohorts below, but ABC4D and ARISES
# are proprietary (see data_pipeline/DATASET.md §1). Only HUPA-UCM and T1D-UOM
# are reproducible from public data, so override the cohort list to run with
# whatever you actually have — --num-clients is derived from it automatically:
#   COHORTS="HUPA-UCM T1D-UOM" ./run_experiment.sh            # public-data-only run
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${METHOD:-mldg}"
SEED="${SEED:-42}"
CONFIG="$ROOT_DIR/config_${METHOD}_seed${SEED}.yaml"
PORT="${PORT:-8088}"
SERVER_URL="http://127.0.0.1:${PORT}"
LOG_DIR="$ROOT_DIR/logs/${METHOD}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"

# Validate everything BEFORE creating any directories, so a bad invocation
# leaves no stray empty log dirs behind.
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found: $CONFIG"
  echo "       Available methods: fedavg fedprox mldg apfl apfl_decoupled ditto"
  echo "       Available seeds:   42 43 44"
  exit 1
fi

SPLITS="$ROOT_DIR/data_output/metabonet_splits"
# Space-separated override, e.g. COHORTS="HUPA-UCM T1D-UOM"
read -r -a COHORTS <<< "${COHORTS:-HUPA-UCM ABC4D ARISES T1D-UOM}"

MISSING=()
for c in "${COHORTS[@]}"; do
  [ -d "$SPLITS/$c" ] || MISSING+=("$c")
done
if [ ${#MISSING[@]} -gt 0 ]; then
  echo "ERROR: missing cohort director$([ ${#MISSING[@]} -eq 1 ] && echo y || echo ies) under $SPLITS:"
  for c in "${MISSING[@]}"; do
    case "$c" in
      ABC4D|ARISES)
        echo "  - $c  (PROPRIETARY — not produced by this release; obtain from the study owners,"
        echo "         or drop it: COHORTS=\"HUPA-UCM T1D-UOM\" $0)" ;;
      *)
        echo "  - $c  (build it with data_pipeline/build_metabonet.py, then pack_metabonet_segments.py)" ;;
    esac
  done
  exit 1
fi

mkdir -p "$LOG_DIR"
echo "=== federated trial — method=$METHOD seed=$SEED, ${#COHORTS[@]} clients on $SERVER_URL ==="
echo "    config: $CONFIG"
echo "    logs:   $LOG_DIR"
cd "$ROOT_DIR"

python3 fl_server.py --config "$CONFIG" --host 127.0.0.1 --port "$PORT" \
    --num-clients "${#COHORTS[@]}" > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "  server pid=$SERVER_PID"

CLIENT_PIDS=()
cleanup() {
  echo; echo "=== shutting down ==="
  kill "$SERVER_PID" "${CLIENT_PIDS[@]}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM
sleep 8

for cohort in "${COHORTS[@]}"; do
  python3 fl_client.py --config "$CONFIG" \
      --source "$SPLITS/$cohort" --server "$SERVER_URL" \
      --client-id "$cohort" > "$LOG_DIR/client_${cohort}.log" 2>&1 &
  CLIENT_PIDS+=($!)
  echo "  client pid=${CLIENT_PIDS[-1]} cohort=$cohort"
done

echo; echo "Following server log (Ctrl-C to stop)..."
echo "------------------------------------------------------------------"
tail -f "$LOG_DIR/server.log" &
TAIL_PID=$!
CLIENT_PIDS+=($TAIL_PID)

for pid in "${CLIENT_PIDS[@]}"; do
  [ "$pid" = "$TAIL_PID" ] && continue
  wait "$pid" 2>/dev/null || true
done

echo; echo "------------------------------------------------------------------"
echo "Done. Results in output_${METHOD}_s${SEED}/ (per config):"
echo "  round_summary.csv           per-round avg val MSE/RMSE + best flag"
echo "  round_client_metrics.csv    per-round per-client train loss + val"
echo "  best_model_test_metrics.csv per-cohort test metrics on the deployed model"
echo "  best_global_model.pt        server-side FedAvg w-path checkpoint."
echo "                              For apfl/apfl_decoupled/ditto this is NOT the"
echo "                              evaluated model — those arms deploy a per-client"
echo "                              personal/mixture model that never leaves the client."
echo "  best_global_model_meta.txt  best round + its avg val MSE"
