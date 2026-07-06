#!/usr/bin/env bash
# Submit 7-node hybrid FedAvg: Llama-150M + QSGD on global gRPC.
#
# Requires (export before run):
#   OMNIFED_C4_DISK, OMNIFED_TOKENIZER_DIR, OMNIFED_LLAMA_WEIGHTS
#
# Usage:
#   QSGD_BIT_WIDTH=4 ./test_scripts/frontier_hybrid_llama150m_7_qsgd.sh
set -euo pipefail

REPO="${OMNIFED_REPO:-/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT}"
ACCOUNT="${SLURM_ACCOUNT:-gen150}"
BIT_WIDTH="${QSGD_BIT_WIDTH:-4}"

: "${OMNIFED_C4_DISK:?Set OMNIFED_C4_DISK}"
: "${OMNIFED_TOKENIZER_DIR:?Set OMNIFED_TOKENIZER_DIR}"
: "${OMNIFED_LLAMA_WEIGHTS:?Set OMNIFED_LLAMA_WEIGHTS}"

cd "${REPO}"
export PYTHONPATH="${REPO}"
export PYEXE="${PYEXE:-${CONDA_PREFIX}/bin/python}"

./main.sh --config-name test_hybrid_layout_fedavg_llama150m_grpc_qsgd \
  overwrite=true \
  engine.mode=slurm \
  global_rounds=2 \
  engine.hybrid.global_compression.enabled=true \
  engine.hybrid.global_compression.scheme=qsgd \
  engine.hybrid.global_compression.bit_width="${BIT_WIDTH}" \
  slurm.account="${ACCOUNT}" \
  slurm.partition=batch \
  slurm.time=01:35:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null \
  slurm.exclusive=true
