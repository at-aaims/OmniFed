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

install_lxc() {
    case "$OS" in
        ubuntu)
            sudo apt update
            sudo apt install -y lxc lxc-utils lxc-templates
            ;;
        centos|rhel)
            sudo yum install -y epel-release
            sudo yum install -y lxc lxc-libs lxc-templates
            ;;
        fedora)
            sudo dnf install -y lxc lxc-libs lxc-templates
            ;;
        arch)
            sudo pacman -S lxc lxc-templates
            ;;
    esac
    echo "installed lxc successfully."
}

install_lxc
sudo systemctl enable lxc
sudo systemctl start lxc
lxc-info --version