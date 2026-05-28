#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; switch to repo root (script is in test_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------------------------------------------------------------
# Quick hybrid / comm tests
# - **Logs + data for this launcher:** each attempt uses
#     ``$OMNIFED_HYBRID_DATA_DIR/hybrid_runs/<run_id>/``
#   so ``g0/``..``gN/`` logs do not append across runs. Override id:
#     ``--run-id mytest`` or ``export OMNIFED_HYBRID_RUN_ID=mytest``.
#   Wipe that folder before launch:
#     ``--fresh`` or ``export OMNIFED_HYBRID_TRUNCATE_LOGS=1``.
# - **Steps:** ``export EPOCHS=1 MAX_STEPS=32`` (see omega_launch).
# - **Slurm:** Phase A Step 4 script: ``test_scripts/slurm_frontier/hybrid_comm_smoke.slurm``.
# -----------------------------------------------------------------------------

# -----------------------
# Parse command line args
# -----------------------
usage() {
  echo "Usage: $0 --config <config_file_name> [--run-id <id>] [--fresh]"
  echo "Example: $0 --config built_symmetric_2x3.yaml --fresh"
  echo "  OMNIFED_HYBRID_DATA_DIR   Base directory (default: .../flora_test)"
  echo "  OMNIFED_HYBRID_RUN_ID     Run id (default: timestamp)"
  echo "  OMNIFED_HYBRID_TRUNCATE_LOGS=1   Same as --fresh"
  exit 1
}

CONFIG_FILE=""
RUN_ID_CLI=""
FRESH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID_CLI="$2"
      shift 2
      ;;
    --fresh)
      FRESH=1
      shift
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -z "${CONFIG_FILE}" ]]; then
  echo "Error: --config is required"
  usage
fi

topology_name="${CONFIG_FILE%.yaml}"
hydra_topology_cfg="${REPO_ROOT}/conf_hybrid/topology/${topology_name}.yaml"

if [[ ! -f "${hydra_topology_cfg}" ]]; then
  echo "Error: Hydra topology config not found: ${hydra_topology_cfg}"
  exit 1
fi

# -----------------------
# Read world_size from Hydra-composed config
# -----------------------
worldsize="$(python3 -u -m src.flora.test.hydra_world_size --config "${CONFIG_FILE}")"

if [[ -z "${worldsize}" ]]; then
  echo "Error: Could not read world_size from Hydra config: ${CONFIG_FILE}"
  exit 1
fi

echo "Using Hydra topology config: ${hydra_topology_cfg}"
echo "World size from Hydra config: ${worldsize}"

# -----------------------
# Run-isolated work dir (stdout + Trainer logs + cifar cache under this tree)
# -----------------------
DATA_ROOT="${OMNIFED_HYBRID_DATA_DIR:-/home/shruti/omnifed_data/flora_test/}"
RUN_ID="${RUN_ID_CLI:-${OMNIFED_HYBRID_RUN_ID:-$(date +%Y%m%d_%H%M%S)}}"
RUN_ROOT="${DATA_ROOT%/}/hybrid_runs/${RUN_ID}"

if [[ "${FRESH}" == "1" ]] || [[ "${OMNIFED_HYBRID_TRUNCATE_LOGS:-}" == "1" ]]; then
  echo "[generic_hybrid_comm.sh] Removing previous run tree: ${RUN_ROOT}"
  rm -rf "${RUN_ROOT}"
fi
mkdir -p "${RUN_ROOT}"

echo "[generic_hybrid_comm.sh] run_id=${RUN_ID}"
echo "[generic_hybrid_comm.sh] run_root=${RUN_ROOT}"

dir="${RUN_ROOT}"
bsz=32
commfreq=7
backend="gloo"
model="resnet18"
dataset="cifar10"

# Optional: limit epochs/steps for fast comm checks (empty = omega_launch defaults).
EPOCHS="${EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-}"

# -----------------------
# Cleanup handling
# -----------------------
pids=()

cleanup() {
  echo ""
  echo "[generic_hybrid_comm.sh] Cleaning up ${#pids[@]} worker(s)..."

  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "  -> SIGTERM ${pid}"
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  sleep 1

  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      echo "  -> SIGKILL ${pid}"
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

trap cleanup INT TERM EXIT

# -----------------------
# Launch ranks
# -----------------------
extra_args=()
if [[ -n "${EPOCHS}" ]]; then
  extra_args+=(--epochs="${EPOCHS}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  extra_args+=(--max-steps="${MAX_STEPS}")
fi

for ((globalrank=0; globalrank<worldsize; globalrank++)); do

  echo "###### launching global_rank=${globalrank}"
  mkdir -p "${dir}/g${globalrank}"
  logf="${dir}/g${globalrank}/stdout.log"

  python3 -u -m src.flora.test.omega_launch_hybridcomm \
    --config="${CONFIG_FILE}" \
    --dir="${dir}" --bsz="${bsz}" --global-rank="${globalrank}" \
    --comm-freq="${commfreq}" --backend="${backend}" \
    --model="${model}" --dataset="${dataset}" \
    --train-dir="${dir}" --test-dir="${dir}" \
    "${extra_args[@]}" \
    >>"${logf}" 2>&1 &

  pid=$!
  pids+=("${pid}")

  echo "Spawned python PID ${pid}; log ${logf}; sleeping 3 seconds..."
  sleep 3
done

echo "Tip: tail -f ${dir}/g*/stdout.log"
wait
