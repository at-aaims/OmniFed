#!/usr/bin/env bash
# Llama ~400M hybrid, 7 nodes, auto-chain until global_rounds=5 (Frontier / gen150).
set -euo pipefail

export OMNIFED_REPO="${OMNIFED_REPO:-/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT}"
export CKPT_ROOT="${CKPT_ROOT:-/lustre/orion/gen150/scratch/shruti2395/omnifed_checkpoints}"
export EXP_ID="${EXP_ID:-llama400_7_autochain_v1}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-gen150}"
export PYEXE="${PYEXE:-/ccs/home/shruti2395/.conda/envs/pytorch_rocm/bin/python}"

export OMNIFED_C4_DISK="${OMNIFED_C4_DISK:-/lustre/orion/gen150/scratch/shruti2395/omnifed_data/allenai_c4}"
export OMNIFED_TOKENIZER_DIR="${OMNIFED_TOKENIZER_DIR:-/lustre/orion/gen150/scratch/shruti2395/omnifed_data/tokenizer_Mistral-7B-v0.1}"
export OMNIFED_LLAMA400_WEIGHTS="${OMNIFED_LLAMA400_WEIGHTS:-/lustre/orion/gen150/scratch/shruti2395/omnifed_data/Llama-3.2-400M-Amharic}"

cd "${OMNIFED_REPO}"
export PYTHONPATH="${OMNIFED_REPO}"
mkdir -p "${CKPT_ROOT}/${EXP_ID}"

exec ./scripts/hybrid_checkpoint_autochain.sh -- ./main.sh \
  --config-name test_hybrid_layout_fedavg_llama400m \
  overwrite=true \
  engine.mode=slurm \
  global_rounds=5 \
  slurm.job_name=llama400_7_autochain \
  slurm.account="${SLURM_ACCOUNT}" \
  slurm.partition=batch \
  slurm.time=01:59:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null \
  slurm.exclusive=true
