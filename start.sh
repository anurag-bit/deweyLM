#!/bin/bash
# Quick Start Script for Dewey RAG System
# Run this script to get started quickly

echo "📚 Dewey Classification RAG System - Quick Start"
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found"

# Check if environment exists
if conda info --envs | grep -q "dewey-rag-server"; then
    echo "✅ Environment 'dewey-rag-server' already exists"
else
    echo "🚀 Creating conda environment..."
    conda env create -f environment_server.yml
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create environment"
        exit 1
    fi
    echo "✅ Environment created successfully"
fi

echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dewey-rag-server

echo "📦 Installing additional packages..."
pip install streamlit plotly altair

echo "🚀 Launching Dewey RAG System..."
python launcher.py