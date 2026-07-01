#!/usr/bin/env bash
# Submit 7-node hybrid FedAvg: CIFAR-10 + ResNet-18 with QSGD on global gRPC.
#
# Usage:
#   QSGD_BIT_WIDTH=4 ./test_scripts/frontier_hybrid_cifar10_resnet18_7_qsgd.sh
#   QSGD_BIT_WIDTH=2 ./test_scripts/frontier_hybrid_cifar10_resnet18_7_qsgd.sh   # 4 levels
#   QSGD_BIT_WIDTH=8 ./test_scripts/frontier_hybrid_cifar10_resnet18_7_qsgd.sh   # 256 levels
#
# Defaults: global_rounds=5, slurm.time=02:00:00 (Frontier batch max is 120 min).
set -euo pipefail

REPO="${OMNIFED_REPO:-/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT}"
CIFAR_ROOT="${OMNIFED_CIFAR10_ROOT:-/lustre/orion/gen150/scratch/shruti2395/omnifed_data/cifar10}"
ACCOUNT="${SLURM_ACCOUNT:-gen150}"
BIT_WIDTH="${QSGD_BIT_WIDTH:-4}"

cd "${REPO}"
export PYTHONPATH="${REPO}"
export PYEXE="${PYEXE:-${CONDA_PREFIX}/bin/python}"

./main.sh --config-name test_hybrid_layout_fedavg_cifar10_resnet18_grpc_qsgd \
  overwrite=true \
  engine.mode=slurm \
  global_rounds=5 \
  engine.hybrid.global_compression.enabled=true \
  engine.hybrid.global_compression.scheme=qsgd \
  engine.hybrid.global_compression.bit_width="${BIT_WIDTH}" \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root="${CIFAR_ROOT}" \
  datamodule.eval.dataset.root="${CIFAR_ROOT}" \
  slurm.account="${ACCOUNT}" \
  slurm.partition=batch \
  slurm.time=02:00:00 \
  slurm.nodes=7 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null \
  slurm.exclusive=true
