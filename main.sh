#!/bin/bash

# =========================================

clear

set -euo pipefail  # Exit on error, undefined variable, or failed command in a pipeline
set -x              # Print each command before executing it (for debugging)

# =========================================
# Set debugging environment variables

export PYTHONUNBUFFERED=1              # Ensure Python output isn't buffered

# PyTorch Distributed debugging and settings
# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Enable detailed debugging output for torch.distributed
# export TORCH_CPP_LOG_LEVEL=INFO        # Get more detailed C++ logs
# export GLOO_SOCKET_IFNAME=lo           # Force Gloo to use loopback interface
# export NCCL_DEBUG=WARN                 # NCCL debugging (for GPU communication)

# Ray settings
export RAY_DEDUP_LOGS=0                # Show all logs, don't deduplicate

# Hydra settings
export HYDRA_FULL_ERROR=1              # Show full error trace for Hydra

# =========================================

rm -rf ./outputs

# =========================================

# python -u main.py --config-name test_mnist_torchdist "$@"

python -u main.py --config-name test_mnist_grpc "$@"

# python -u main.py --config-name test_mnist_grpc "$@" 2>&1 | tee main.log
