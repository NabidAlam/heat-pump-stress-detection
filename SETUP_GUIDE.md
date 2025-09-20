# Energy Publication Project - Setup Guide

This guide will help you set up the complete environment for the Energy Publication project, including CUDA support for transformer training.

## Quick Setup (Recommended)

### Option 1: Automated Setup
```bash
# Run the automated setup script
python setup_environment.py
```

### Option 2: Manual Setup
```bash
# Create conda environment from yml file
conda env create -f environment.yml

# Activate environment
conda activate energy-publication

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: 16GB+ recommended (32GB+ for full dataset)
- **Storage**: 10GB+ free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software Requirements
- **Anaconda/Miniconda**: [Download here](https://docs.conda.io/en/latest/miniconda.html)
- **NVIDIA Driver**: [Download here](https://www.nvidia.com/drivers/) (for GPU support)
- **CUDA Toolkit**: Included in conda environment

## Detailed Setup Instructions

### 1. Install Anaconda/Miniconda
```bash
# Download and install from official website
# https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
```

### 2. Clone/Download Project
```bash
# If using git
git clone <repository-url>
cd energy-publication

# Or download and extract the project files
```

### 3. Create Environment
```bash
# Method 1: Using environment.yml (recommended)
conda env create -f environment.yml

# Method 2: Using requirements.txt
conda create -n energy-publication python=3.9
conda activate energy-publication
pip install -r requirements.txt
```

### 4. Activate Environment
```bash
# Activate the environment
conda activate energy-publication

# Verify activation
conda info --envs
```

### 5. Verify Installation
```bash
# Test core packages
python -c "import numpy, pandas, sklearn; print('Core packages OK')"

# Test PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"

# Test CUDA (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test transformers
python -c "import transformers; print(f'Transformers {transformers.__version__} OK')"
```

## Running the Analysis

### 1. Basic Analysis (Current Implementation)
```bash
# Run focused analysis with existing models
python scripts/run_focused_analysis.py --scope pan_european

# Generate figures
python scripts/generate_paper_figures.py
```

### 2. Transformer Analysis (Memory-Efficient)
```bash
# Run transformer analysis with memory optimization
python scripts/run_transformer_analysis_optimized.py --model_type tft --sample_size 50000

# For full dataset (requires 32GB+ RAM)
python scripts/run_transformer_analysis_optimized.py --model_type tft --sample_size 100000
```

### 3. Full Transformer Analysis (High Memory)
```bash
# Only run if you have 32GB+ RAM
python scripts/run_transformer_analysis.py --model_type tft --sequence_length 168
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 3.5+
- **VRAM**: 4GB+ recommended (8GB+ for full dataset)
- **Driver**: Latest NVIDIA driver

### Verify GPU Setup
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### GPU Memory Optimization
If you encounter GPU memory issues:
```bash
# Reduce batch size in training scripts
# Edit scripts/run_transformer_analysis_optimized.py
# Change batch_size from 16 to 8 or 4

# Use smaller model architecture
# Edit src/models/transformer_models.py
# Reduce d_model from 64 to 32
```

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

#### 2. Memory Issues
```bash
# Use memory-efficient version
python scripts/run_transformer_analysis_optimized.py --sample_size 30000

# Reduce sequence length
python scripts/run_transformer_analysis_optimized.py --sequence_length 84
```

#### 3. Package Conflicts
```bash
# Create fresh environment
conda env remove -n energy-publication
conda env create -f environment.yml
```

#### 4. Slow Training
```bash
# Use GPU acceleration
python -c "import torch; print(torch.cuda.is_available())"

# Reduce model complexity
# Edit model parameters in scripts
```

### Performance Tips

#### For CPU-Only Systems
- Use smaller sample sizes (--sample_size 20000)
- Reduce sequence length (--sequence_length 84)
- Use simpler model architectures

#### For GPU Systems
- Use full sample sizes for better performance
- Increase batch size if memory allows
- Use mixed precision training (future enhancement)

## Expected Performance

### System Requirements by Dataset Size
| Dataset Size | RAM Required | GPU VRAM | Training Time |
|--------------|--------------|----------|---------------|
| 20K samples  | 8GB         | 2GB      | 10-30 min     |
| 50K samples  | 16GB        | 4GB      | 30-60 min     |
| 100K samples | 32GB        | 8GB      | 1-2 hours     |
| Full (131K)  | 64GB        | 16GB     | 2-4 hours     |

### Performance Improvements
- **Transformer vs XGBoost**: Expected 15-25% improvement in F1-score
- **GPU vs CPU**: 5-10x faster training
- **Memory optimization**: Enables training on smaller systems

## Next Steps

After successful setup:

1. **Run Basic Analysis**: Start with the focused analysis
2. **Test Transformer**: Run memory-efficient transformer analysis
3. **Generate Figures**: Create publication-ready figures
4. **Compare Results**: Analyze transformer vs baseline performance
5. **Write Paper**: Use results for 5-page paper

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Verify system requirements
3. Try memory-efficient versions
4. Check CUDA installation (for GPU users)

## Success Indicators

You'll know setup is successful when:
- All packages import without errors
- CUDA is available (if GPU present)
- Basic analysis runs successfully
- Transformer training starts without memory errors
- Figures generate correctly


