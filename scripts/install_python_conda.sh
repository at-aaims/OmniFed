#!/bin/bash

# Detect Linux Distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Unsupported OS"
    exit 1
fi

# Function to install Python 3.9
install_python() {
    case "$OS" in
        ubuntu|debian)
            sudo apt update && sudo apt upgrade -y
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            sudo apt install -y python3.9 python3.9-venv python3.9-dev
            ;;
        centos|rhel)
            sudo yum install -y gcc gcc-c++ make
            sudo yum install -y python39 python39-devel python39-pip
            ;;
        fedora)
            sudo dnf install -y python3.9 python3.9-devel
            ;;
        arch)
            sudo pacman -Syu --noconfirm python39
            ;;
        *)
            echo "Unsupported OS: $OS"
            exit 1
            ;;
    esac
    echo "Python 3.9 installed successfully."
}

# Function to install Miniconda
install_conda() {
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    echo "Miniconda installed successfully."
}

# Run installation functions
install_python
install_conda

# Verify installation
python3.9 --version
conda --version

echo "Installation completed successfully!"
