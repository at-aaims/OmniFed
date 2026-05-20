#!/usr/bin/env bash
set -euo pipefail

# Run from anywhere; switch to repo root (script is in test_scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -----------------------
# User-configurable params
# -----------------------
dir="/home/shruti/omnifed_data/flora_test/"
bsz=32
worldsize=7
commfreq=7
backend="gloo"
model="resnet18"
dataset="cifar10"

# Map global_rank -> local_rank
# Indices 0..worldsize-1
#localranks=(-1 0 0 1 2 1 2)

# adding config from yaml
config="${REPO_ROOT}/config/try1_hybrid_topo.yaml"

EPOCHS="${EPOCHS:-}"
MAX_STEPS="${MAX_STEPS:-}"
extra_args=()
[[ -n "${EPOCHS}" ]] && extra_args+=(--epochs="${EPOCHS}")
[[ -n "${MAX_STEPS}" ]] && extra_args+=(--max-steps="${MAX_STEPS}")

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
  #localrank="${localranks[$globalrank]:--1}"

  echo "###### launching global_rank=${globalrank}"

  mkdir -p "${dir}/g${globalrank}"
  logf="${dir}/g${globalrank}/stdout.log"

  python3 -u -m src.flora.test.omega_launch_hybridcomm \
    --config="${config}" \
    --dir="${dir}" --bsz="${bsz}" --global-rank="${globalrank}" \
    --comm-freq="${commfreq}" --backend="${backend}" \
    --model="${model}" --dataset="${dataset}" \
    --train-dir="${dir}" --test-dir="${dir}" \
    "${extra_args[@]}" \
    >>"${logf}" 2>&1 &
    
#> "${dir}/g${globalrank}/stdout.log" 2>&1 &

  pid=$!
  pids+=("$pid")

  echo "Spawned PID ${pid}; sleeping 3 seconds..."
  sleep 3
done

# Keep the script attached to workers; Ctrl+C will trigger trap and cleanup.
wait
