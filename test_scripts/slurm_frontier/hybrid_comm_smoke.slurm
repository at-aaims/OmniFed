#!/bin/bash
#SBATCH -A gen150
#SBATCH -J hybrid_smoke
#SBATCH -p batch
#SBATCH -t 00:15:00
#SBATCH -N 7
#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH -o outputs/hybrid_smoke_slurm/%x-%j.out
#SBATCH -e outputs/hybrid_smoke_slurm/%x-%j.err
#
# Phase A Step 4: multi-node hybrid smoke (torch per facility + one gRPC round).
#
# Intended for OLCF Frontier (or similar): login node + modules + GPU nodes.
# Local workstations often have incomplete Slurm; use instead:
#   bash test_scripts/hybrid_comm_smoke_local.sh
#   bash test_scripts/generic_hybrid_comm.sh --config built_symmetric_2x3.yaml
# Submit script name: hybrid_comm_smoke.slurm  (not ...snope...).
#
# Submit from your OmniFed_VT repo root (paths above are relative to that cwd):
#   export OMNIFED_REPO=$PWD   # optional; batch script cds here
#   export PYEXE=/ccs/home/USER/.conda/envs/pytorch_rocm/bin/python
#   sbatch test_scripts/slurm_frontier/hybrid_comm_smoke.slurm
#
# Assumes compact Slurm mapping: rank i maps to host i in
#   scontrol show hostnames $SLURM_NODELIST
# (typical with --ntasks-per-node=1 across -N 7).
#
# Optional: HYBRID_TOPO_CONFIG, HYBRID_SMOKE_BACKEND (gloo=CPU default, nccl=GPU/RCCL),
#   HYBRID_SMOKE_SERVER_WAIT_SEC

set -euo pipefail

REPO="${OMNIFED_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO}"
export PYTHONPATH="${REPO}"

: "${PYEXE:?Set PYEXE to compute Python, e.g. .../pytorch_rocm/bin/python}"

module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

mapfile -t _HOSTS < <(scontrol show hostnames "${SLURM_NODELIST}")
if [[ "${#_HOSTS[@]}" -lt 3 ]]; then
  echo "Need at least 3 nodes (server + 2 facility leaders); got ${#_HOSTS[@]}"
  exit 1
fi

export OMNIFED_HYBRID_RPC_ADDR="${_HOSTS[0]}"
export OMNIFED_HYBRID_FACILITY_MPI_ADDRS="${_HOSTS[1]},${_HOSTS[2]}"
export HYBRID_SMOKE_SERVER_WAIT_SEC="${HYBRID_SMOKE_SERVER_WAIT_SEC:-120}"
export HYBRID_SMOKE_RANK_STAGGER_SEC="${HYBRID_SMOKE_RANK_STAGGER_SEC:-2}"
export HYBRID_TOPO_CONFIG="${HYBRID_TOPO_CONFIG:-built_symmetric_2x3.yaml}"
export HYBRID_SMOKE_BACKEND="${HYBRID_SMOKE_BACKEND:-gloo}"

mkdir -p "${REPO}/outputs/hybrid_smoke_slurm"

echo "REPO=${REPO} PYEXE=${PYEXE}"
echo "CFG=${HYBRID_TOPO_CONFIG} BACKEND=${HYBRID_SMOKE_BACKEND}"
echo "RPC=${OMNIFED_HYBRID_RPC_ADDR}  FAC_MPI=${OMNIFED_HYBRID_FACILITY_MPI_ADDRS}"

WRAP="${REPO}/test_scripts/slurm_frontier/run_hybrid_smoke_one_task.sh"
chmod +x "${WRAP}"

srun --export=ALL -l -u "${WRAP}"

echo "hybrid_comm_smoke job completed."
