#!/bin/bash

git submodule add https://github.com/pytorch/pytorch.git ../third_party/pytorch
cd ../third_party/pytorch
git checkout tags/v2.6.0
git submodule update --init --recursive
conda install cmake ninja
pip install -r requirements.txt
pip install mkl-static mkl-include
make triton

# Check if ROCm is installed and a GPU is detected
if command -v rocminfo &> /dev/null; then
    if rocminfo | grep -q "GPU ID"; then
        echo "AMD ROCm device detected"
        python tools/amd_build/build_amd.py
    fi
fi

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-'$(dirname $(which conda))/../'}:${CMAKE_PREFIX_PATH}"
python setup.py develop