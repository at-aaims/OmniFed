#!/usr/bin/env bash
# Login-node wrapper: submit hybrid Slurm jobs until manifest status is ``complete``.
#
# Required env:
#   OMNIFED_REPO, CKPT_ROOT, EXP_ID, PYEXE (or conda active)
# Required args: all ``./main.sh`` Hydra overrides after ``--`` (must include
#   slurm.checkpoint_dir, slurm.experiment_id, global_rounds, engine.mode=slurm).
#
# Example:
#   export CKPT_ROOT=.../omnifed_checkpoints EXP_ID=llama400_7_chain_v1
#   ./scripts/hybrid_checkpoint_autochain.sh -- ./main.sh --config-name test_hybrid_layout_fedavg_llama400m ...
#
set -euo pipefail

MAX_CHAINS="${OMNIFED_MAX_CHAINS:-30}"
POLL_SEC="${OMNIFED_CHAIN_POLL_SEC:-60}"
REPO="${OMNIFED_REPO:?set OMNIFED_REPO}"
CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT}"
EXP_ID="${EXP_ID:?set EXP_ID}"
EXP_DIR="${CKPT_ROOT}/${EXP_ID}"

if [[ "${1:-}" != "--" ]]; then
  echo "Usage: CKPT_ROOT=... EXP_ID=... $0 -- ./main.sh [hydra overrides...]" >&2
  exit 1
fi
shift

cd "${REPO}"
export PYTHONPATH="${REPO}"
mkdir -p "${EXP_DIR}"

RESUME=false
if "${PYEXE:-python}" -c "
from src.omnifed.checkpoint.hybrid_round_checkpoint import load_manifest
m = load_manifest('${EXP_DIR}')
raise SystemExit(0 if m and m.get('status') == 'in_progress' else 1)
"; then
  RESUME=true
  echo "[autochain] found in_progress manifest at ${EXP_DIR}; starting with slurm.resume=true"
fi
CHAIN=0

wait_for_job() {
  local job_id="$1"
  echo "[autochain] waiting for job ${job_id} (poll ${POLL_SEC}s)..."
  while squeue -j "${job_id}" -h 2>/dev/null | grep -q .; do
    sleep "${POLL_SEC}"
  done
  echo "[autochain] sacct summary for job ${job_id}:"
  sacct -j "${job_id}" --format=JobID,JobName,State,ExitCode,Elapsed -P 2>/dev/null | head -20 || true

  local state exitcode
  state=$(sacct -j "${job_id}" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ' || true)
  exitcode=$(sacct -j "${job_id}" --format=ExitCode -n -X 2>/dev/null | head -1 | tr -d ' ' || true)
  echo "[autochain] job ${job_id} final state: ${state:-UNKNOWN} ExitCode=${exitcode:-?}"

  local log_hint
  log_hint=$(find "${REPO}/outputs" -name "slurm-${job_id}.out" 2>/dev/null | head -1 || true)
  if [[ -n "${log_hint}" ]]; then
    echo "[autochain] slurm stdout: ${log_hint}"
    echo "[autochain] last 40 lines:"
    tail -40 "${log_hint}" || true
    echo "[autochain] errors in .out / .err:"
    grep -iE 'traceback|fatal|error|FileNotFound|ModuleNotFound|ValueError|RuntimeError' \
      "${log_hint}" "${log_hint%.out}.err" 2>/dev/null | tail -30 || true
  else
    echo "[autochain] log not found yet; try: find ${REPO}/outputs -name 'slurm-${job_id}.*'"
  fi

  if [[ "${state}" != *COMPLETED* ]]; then
    if "${PYEXE:-python}" -c "
from src.omnifed.checkpoint.hybrid_round_checkpoint import load_manifest
m = load_manifest('${EXP_DIR}')
raise SystemExit(0 if m and m.get('status') == 'in_progress' else 1)
"; then
      echo "[autochain] job ended as ${state:-UNKNOWN} (ExitCode=${exitcode:-?}) but manifest is in_progress — chaining will resubmit with resume=true"
      return 0
    fi
    echo "[autochain] job did not COMPLETE and no resumable checkpoint; stopping chain." >&2
    echo "[autochain] fix the failure, then resume manually with slurm.resume=true or re-run this wrapper." >&2
    exit 1
  fi
}

parse_job_id() {
  local log_file="$1"
  grep -oE 'Submitted batch job [0-9]+' "${log_file}" | tail -1 | awk '{print $4}'
}

while (( CHAIN < MAX_CHAINS )); do
  CHAIN=$((CHAIN + 1))
  echo "========== autochain iteration ${CHAIN}/${MAX_CHAINS} resume=${RESUME} =========="

  LOG=$(mktemp)
  set +e
  "$@" \
    slurm.resume="${RESUME}" \
    slurm.dependency_singleton=true \
    slurm.checkpoint_dir="${CKPT_ROOT}" \
    slurm.experiment_id="${EXP_ID}" \
    2>&1 | tee "${LOG}"
  SUBMIT_RC=${PIPESTATUS[0]}
  set -e

  if (( SUBMIT_RC != 0 )); then
    echo "[autochain] main.sh submit failed (rc=${SUBMIT_RC})" >&2
    rm -f "${LOG}"
    exit "${SUBMIT_RC}"
  fi

  JOB_ID=$(parse_job_id "${LOG}")
  rm -f "${LOG}"
  if [[ -z "${JOB_ID}" ]]; then
    echo "[autochain] could not parse Slurm job id from submit output" >&2
    exit 1
  fi
  echo "[autochain] submitted job ${JOB_ID}"

  wait_for_job "${JOB_ID}"

  if "${PYEXE:-python}" -c "
from src.omnifed.checkpoint.hybrid_round_checkpoint import is_training_complete
import sys
sys.exit(0 if is_training_complete('${EXP_DIR}') else 1)
"; then
    echo "[autochain] training complete: ${EXP_DIR}/manifest.json"
    cat "${EXP_DIR}/manifest.json"
    exit 0
  fi

  echo "[autochain] manifest still in_progress; will resubmit with slurm.resume=true"
  RESUME=true
done

echo "[autochain] exceeded MAX_CHAINS=${MAX_CHAINS} without complete status" >&2
exit 1
