#!/bin/bash

echo "🚀 Setting up LoRA Training Environment..."

# Create virtual environment
python -m venv lora_env
source lora_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

echo "✅ Setup complete! Activate environment with: source lora_env/bin/activate"