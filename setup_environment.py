#!/usr/bin/env python3
"""
Environment Setup Script for Energy Publication Project
======================================================

This script sets up the conda environment with all required dependencies
including CUDA support for transformer training.
"""

import subprocess
import sys
import os
from pathlib import Path
import platform


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"{description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed")
        print(f"Error: {e.stderr}")
        return False


def check_cuda_availability():
    """Check if CUDA is available on the system."""
    print("\nChecking CUDA availability...")
    
    # Check NVIDIA driver
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("NVIDIA driver detected")
            print(f"NVIDIA-SMI output:\n{result.stdout}")
            return True
        else:
            print("NVIDIA driver not found")
            return False
    except FileNotFoundError:
        print("nvidia-smi not found - NVIDIA driver not installed")
        return False


def create_conda_environment():
    """Create conda environment from environment.yml."""
    print("\nSetting up conda environment...")
    
    # Check if conda is available
    if not run_command("conda --version", "Checking conda installation"):
        print("Conda not found. Please install Anaconda or Miniconda first.")
        print("Download from: https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Create environment from yml file
    if not run_command("conda env create -f environment.yml", "Creating conda environment"):
        print("Failed to create environment from environment.yml")
        return False
    
    return True


def activate_and_test_environment():
    """Activate environment and test key packages."""
    print("\n🧪 Testing environment setup...")
    
    # Test Python
    test_commands = [
        ("python -c \"import sys; print(f'Python version: {sys.version}')\"", "Testing Python"),
        ("python -c \"import torch; print(f'PyTorch version: {torch.__version__}')\"", "Testing PyTorch"),
        ("python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"", "Testing CUDA support"),
        ("python -c \"import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')\"", "Testing CUDA version"),
        ("python -c \"import pandas as pd; print(f'Pandas version: {pd.__version__}')\"", "Testing Pandas"),
        ("python -c \"import numpy as np; print(f'NumPy version: {np.__version__}')\"", "Testing NumPy"),
        ("python -c \"import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')\"", "Testing Scikit-learn"),
        ("python -c \"import xgboost as xgb; print(f'XGBoost version: {xgb.__version__}')\"", "Testing XGBoost"),
        ("python -c \"import transformers; print(f'Transformers version: {transformers.__version__}')\"", "Testing Transformers"),
    ]
    
    success_count = 0
    for command, description in test_commands:
        if run_command(f"conda run -n energy-publication {command}", description):
            success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{len(test_commands)} tests passed")
    return success_count == len(test_commands)


def create_activation_script():
    """Create activation script for easy environment activation."""
    script_content = """#!/bin/bash
# Energy Publication Project - Environment Activation Script

echo "Activating Energy Publication Environment..."
conda activate energy-publication

echo "Environment activated!"
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")"

echo ""
echo "Ready to run transformer analysis!"
echo "   python scripts/run_transformer_analysis.py --model_type tft"
echo ""
"""
    
    with open("activate_env.sh", "w") as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("activate_env.sh", 0o755)
    
    print("Created activation script: activate_env.sh")


def create_windows_activation_script():
    """Create Windows batch file for environment activation."""
    script_content = """@echo off
REM Energy Publication Project - Environment Activation Script

echo Activating Energy Publication Environment...
call conda activate energy-publication

echo Environment activated!
echo Current directory: %CD%
echo Python version: 
python --version
echo PyTorch CUDA available: 
python -c "import torch; print(torch.cuda.is_available())"

echo.
echo Ready to run transformer analysis!
echo    python scripts/run_transformer_analysis.py --model_type tft
echo.
pause
"""
    
    with open("activate_env.bat", "w") as f:
        f.write(script_content)
    
    print("Created Windows activation script: activate_env.bat")


def main():
    """Main setup function."""
    print("=" * 60)
    print("ENERGY PUBLICATION PROJECT - ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check system
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    # Create environment
    if not create_conda_environment():
        print("Environment setup failed!")
        return False
    
    # Test environment
    if not activate_and_test_environment():
        print("Some tests failed, but environment was created")
    
    # Create activation scripts
    create_activation_script()
    if platform.system() == "Windows":
        create_windows_activation_script()
    
    print("\n" + "=" * 60)
    print("ENVIRONMENT SETUP COMPLETED!")
    print("=" * 60)
    
    print("\nNext Steps:")
    print("1. Activate the environment:")
    if platform.system() == "Windows":
        print("   activate_env.bat")
    else:
        print("   source activate_env.sh")
    print("   # OR manually: conda activate energy-publication")
    
    print("\n2. Run the transformer analysis:")
    print("   python scripts/run_transformer_analysis.py --model_type tft")
    
    print("\n3. Generate paper figures:")
    print("   python scripts/generate_paper_figures.py")
    
    if cuda_available:
        print("\nCUDA is available - GPU acceleration enabled!")
    else:
        print("\nCUDA not available - will use CPU (slower but functional)")
    
    print("\nFor more information, see README.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
