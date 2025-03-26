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

install_singularity_dependencies() {
    case "$OS" in
        ubuntu)
            sudo apt update
            sudo apt install -y build-essential libssl-dev uuid-dev libgpgme-dev squashfs-tools libseccomp-dev \
            wget curl git cryptsetup
            ;;
        centos|rhel)
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y epel-release
            sudo yum install -y golang git wget squashfs-tools cryptsetup libseccomp-devel libgpgme-devel
            ;;
        fedora)
            sudo dnf install -y make git wget squashfs-tools cryptsetup libseccomp-devel libgpgme-devel
            ;;
        arch)
            sudo pacman -S --needed go squashfs-tools cryptsetup
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
    echo "installed all dependencies.."
}

install_singularity_dependencies

export VERSION=1.21.0
wget https://go.dev/dl/go${VERSION}.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go${VERSION}.linux-amd64.tar.gz
export PATH="/usr/local/go/bin:$PATH"
go version

export VERSION=4.0.0  # Check for latest version at https://github.com/apptainer/apptainer/releases
wget https://github.com/apptainer/apptainer/releases/download/v${VERSION}/apptainer-${VERSION}.tar.gz
tar -xzf apptainer-${VERSION}.tar.gz
cd apptainer-${VERSION}
./mconfig && make -C builddir && sudo make -C builddir install
singularity --version
echo "successfully installed singularity."