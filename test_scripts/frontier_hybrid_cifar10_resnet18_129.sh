#!/usr/bin/env bash
# Submit 129-node hybrid FedAvg: CIFAR-10 + ResNet-18 (2×64 trainers + RPC).
set -euo pipefail

REPO="${OMNIFED_REPO:-/lustre/orion/gen150/scratch/shruti2395/OmniFed_VT}"
CIFAR_ROOT="${OMNIFED_CIFAR10_ROOT:-/lustre/orion/gen150/scratch/shruti2395/omnifed_data/cifar10}"
ACCOUNT="${SLURM_ACCOUNT:-gen150}"

cd "${REPO}"
export PYTHONPATH="${REPO}"
export PYEXE="${PYEXE:-${CONDA_PREFIX}/bin/python}"

./main.sh --config-name test_hybrid_layout_fedavg_cifar10_resnet18 \
  overwrite=true \
  engine.mode=slurm \
  global_rounds=10 \
  engine.hybrid.layout.mpi_ranks_per_facility=64 \
  topology.num_clients=128 \
  engine.hybrid.training.dataset_total_clients=128 \
  datamodule.train.dataset.download=false \
  datamodule.eval.dataset.download=false \
  datamodule.train.dataset.root="${CIFAR_ROOT}" \
  datamodule.eval.dataset.root="${CIFAR_ROOT}" \
  slurm.account="${ACCOUNT}" \
  slurm.partition=batch \
  slurm.time=04:00:00 \
  slurm.nodes=129 \
  slurm.ntasks_per_node=1 \
  slurm.cpus_per_task=4 \
  slurm.gpus_per_node=1 \
  slurm.gpus_per_task=1 \
  slurm.gres=null \
  slurm.exclusive=true
