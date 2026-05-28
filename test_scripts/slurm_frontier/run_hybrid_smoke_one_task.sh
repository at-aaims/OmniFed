#!/usr/bin/env bash
# One Slurm task: GPU env shim + hybrid_comm_smoke (SLURM_PROCID from srun).
set -euo pipefail
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" && -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
fi
unset ROCR_VISIBLE_DEVICES
echo "[${SLURM_PROCID:-?}] host=$(hostname)" >&2
exec "${PYEXE:?}" -u -m src.omnifed.hybrid.hybrid_comm_smoke \
  --config "${HYBRID_TOPO_CONFIG:-built_symmetric_2x3.yaml}" \
  --backend "${HYBRID_SMOKE_BACKEND:-gloo}"
