#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; switch to repo root (script is in test_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------
# Parse command line args
# -----------------------
usage() {
  echo "Usage: $0 --config <config_file_name>"
  echo "Example: bash generic_hybrid_comm.sh --config try1_hybrid_topo.yaml"
  exit 1
}

CONFIG_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

if [[ -z "$CONFIG_FILE" ]]; then
  echo "Error: --config is required"
  usage
fi

topology_name="${CONFIG_FILE%.yaml}"
hydra_topology_cfg="${REPO_ROOT}/conf_hybrid/topology/${topology_name}.yaml"

if [[ ! -f "$hydra_topology_cfg" ]]; then
  echo "Error: Hydra topology config not found: $hydra_topology_cfg"
  exit 1
fi

# -----------------------
# Read world_size from Hydra-composed config
# -----------------------
worldsize="$(python3 -u -m src.flora.test.hydra_world_size --config "${CONFIG_FILE}")"

if [[ -z "$worldsize" ]]; then
  echo "Error: Could not read world_size from Hydra config: $CONFIG_FILE"
  exit 1
fi

echo "Using Hydra topology config: $hydra_topology_cfg"
echo "World size from Hydra config: $worldsize"

# -----------------------
# User-configurable params (hardcoded)
# -----------------------
dir="/home/shruti/omnifed_data/flora_test/"
bsz=32
commfreq=7
backend="gloo"
model="resnet18"
dataset="cifar10"

# -----------------------
# Cleanup handling
# -----------------------
pids=()

cleanup() {
  echo ""
  echo "[hybrid_comm.sh] Cleaning up ${#pids[@]} worker(s)..."

  # Graceful stop first
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  -> SIGTERM $pid"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  # Give them a moment
  sleep 1

  # Force kill any remaining
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  -> SIGKILL $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

trap cleanup INT TERM EXIT

# -----------------------
# Launch ranks
# -----------------------
for ((globalrank=0; globalrank<worldsize; globalrank++)); do

  echo "###### launching global_rank=${globalrank}"
  mkdir -p "${dir}/g${globalrank}"

  python3 -u -m src.flora.test.omega_launch_hybridcomm \
    --config="${CONFIG_FILE}" \
    --dir="${dir}" --bsz="${bsz}" --global-rank="${globalrank}" \
    --comm-freq="${commfreq}" --backend="${backend}" \
    --model="${model}" --dataset="${dataset}" \
    --train-dir="${dir}" --test-dir="${dir}" \
    2>&1 | tee "${dir}/g${globalrank}/stdout.log" &

  pid=$!
  pids+=("$pid")

  echo "Spawned PID ${pid}; sleeping 3 seconds..."
  sleep 3
done

# Keep the script attached to workers; Ctrl+C will trigger trap and cleanup.
wait
