#!/usr/bin/env python3
"""
Energy Publication Project Setup Script
Ensures reproducible environment setup
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "results/figures",
        "results/monitoring"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main setup function"""
    print("Setting up Energy Publication Project...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\nCreating project directories...")
    create_directories()
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("Dependency installation failed. Please install manually:")
        print("   pip install -r requirements.txt")
    
    # Verify installation
    print("\nVerifying installation...")
    try:
        import numpy
        import pandas
        import sklearn
        import xgboost
        import matplotlib
        print("All core dependencies are available")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("   Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your raw data files in data/raw/")
    print("2. Run the analysis: python scripts/run_analysis.py --country None")
    print("3. Check results in the results/ directory")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()