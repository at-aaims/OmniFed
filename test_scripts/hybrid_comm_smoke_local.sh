#!/usr/bin/env bash
# Local launcher for src.omnifed.hybrid.hybrid_comm_smoke (no Slurm).
# Staggers ranks similarly to generic_hybrid_comm.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_FILE="${1:-built_symmetric_2x3.yaml}"
LOGDIR="${OMNIFED_HYBRID_SMOKE_LOGDIR:-${REPO_ROOT}/outputs/hybrid_smoke_logs}"
mkdir -p "${LOGDIR}"
worldsize="$(python3 -u -m src.flora.test.hydra_world_size --config "${CONFIG_FILE}")"
backend="${BACKEND:-gloo}"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 1
  for pid in "${pids[@]:-}"; do
    kill -KILL "${pid}" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

for ((r=0; r<worldsize; r++)); do
  python3 -u -m src.omnifed.hybrid.hybrid_comm_smoke \
    --config="${CONFIG_FILE}" \
    --global-rank="${r}" \
    --backend="${backend}" \
    >>"${LOGDIR}/rank_${r}.log" 2>&1 &
  pids+=("$!")
  sleep 2
done
wait
echo "Logs under ${LOGDIR}/"
