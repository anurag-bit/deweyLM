#!/bin/bash
# Server Setup Script for AMD MI300X GPU Server
# This script sets up the environment for processing the Dewey Decimal Classification PDF

echo "Setting up environment for AMD MI300X GPU server..."
echo "Server specs: AMD MI300X, 192GB VRAM, 240GB RAM, 20 vCPUs"

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y build-essential cmake git wget curl

# Install ROCm for AMD GPU support
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install -y rocm-dev rocm-libs

# Install Miniconda if not present
if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
fi

# Create conda environment for the project
conda create -n dewey-rag-server python=3.11 -y
conda activate dewey-rag-server

echo "Installing packages optimized for high-performance processing..."

# Install PyTorch with ROCm support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install additional packages
pip install --upgrade pip
pip install -r requirements_server.txt

echo "Environment setup completed!"
echo "To activate: conda activate dewey-rag-server"