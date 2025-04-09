#!/bin/bash

# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Unsupported OS"
    exit 1
fi

install_mpi() {
    case "$OS" in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
            ;;
        centos|rhel)
            sudo yum install -y epel-release
            sudo yum install -y openmpi openmpi-devel
            echo "export PATH=/usr/lib64/openmpi/bin:\$PATH" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
            source ~/.bashrc
            ;;
        fedora)
            sudo dnf install -y openmpi openmpi-devel
            echo "export PATH=/usr/lib64/openmpi/bin:\$PATH" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
            source ~/.bashrc
            ;;
        arch)
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm openmpi
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
    echo "OpenMPI installation complete!"
    mpirun --version
}

install_mpi_macos(){
    if ! command -v brew &> /dev/null; then
        echo "⚠️ Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        eval "$(/opt/homebrew/bin/brew shellenv)"  # For Apple Silicon
    fi
    brew update
    brew install open-mpi
}

OS_TYPE="$(uname)"
if [ "$OS_TYPE" == "Darwin" ]; then
    echo "🍏 macOS detected"
    install_mpi_macos
fi

install_mpi
install_mpi_macos