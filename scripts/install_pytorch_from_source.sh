#!/bin/bash

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Unsupported OS"
    exit 1
fi

install_openMPI() {
    case "$OS" in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y build-essential cmake libopenmpi-dev
            ;;
        centos|rhel)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y cmake openmpi-devel
            ;;
        fedora)
            sudo dnf install -y cmake openmpi-devel
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
}

git submodule add https://github.com/pytorch/pytorch.git ../third_party/pytorch
cd ../third_party/pytorch
git checkout tags/v2.6.0
git submodule sync
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
pip install mkl-static mkl-include
#make triton

ARG=$1
if [ "$ARG" == "USE_MPI" ]; then
  echop "building pytorch with MPI communication support..."
  install_openMPI
  export USE_MPI=1
  # if using conda
  export CMAKE_PREFIX_PATH=$(dirname $(which conda))/../
else
  echo "building pytorch with Gloo and NCCL communication support..."
fi

# Check if ROCm is installed and a GPU is detected
if command -v rocminfo &> /dev/null; then
    if rocminfo | grep -q "GPU ID"; then
        echo "AMD ROCm device detected, building with support for AMD ROCm..."
        # additional flags
        export USE_ROCM=1
        export USE_NCCL=1  # Enables RCCL (ROCm NCCL equivalent)
        export USE_DISTRIBUTED=1  # Enables distributed training
#        export ROCM_HOME=/opt/rocm
#        export PATH=$ROCM_HOME/bin:$PATH
#        export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
        python tools/amd_build/build_amd.py
    fi
fi

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop