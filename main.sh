#!/bin/bash

# =========================================

echo "###########################################################################"
echo "###########################################################################"
echo

set -euo pipefail # Exit on error, undefined variable, or failed command in a pipeline
set -x            # Print each command before executing it (for debugging)

# to kill test job on mac:  kill -9 $(ps aux | grep main.py | grep -v grep | awk '{print $2}')
# to kill test job on linux systems: kill -s 9 `ps -ef | grep main.py |grep -v grep | awk '{print $2}'`

# =========================================

export OMNIFED_DEBUG=1 # NOTE: currently unused

export PYTHONUNBUFFERED=1 # Immediate Python output
export HYDRA_FULL_ERROR=1 # Full Hydra error traces
export RAY_DEDUP_LOGS=0   # Show all FL node logs

# Also check the yaml configuration files for any additional per-experiment environment variables.

# =========================================

# Compile gRPC protocol buffers
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./src/omnifed/communicator/grpc.proto

# Run the main script
# NOTE: additional arguments passed to the `main.sh` script are forwarded to the Python script
# https://hydra.cc/docs/advanced/hydra-command-line-flags/
python -u main.py "$@"

# python -u main.py --config-name test_mnist_grpc "$@"

# =========================================

# python -u main.py --config-name test_mnist_grpc "$@" 2>&1 | tee main.log
