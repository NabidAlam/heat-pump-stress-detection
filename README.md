# Energy Publication Project

A machine learning project for energy system stress detection using cost-sensitive learning and threshold optimization.

## Key Features

- **Cost-Sensitive Learning**: Advanced approach for handling imbalanced datasets without synthetic data generation
- **Threshold Optimization**: Dynamic threshold adjustment for optimal classification performance
- **🌍 Pan-European Analysis**: Comprehensive analysis across 28 European countries
- **📊 Professional Visualizations**: IEEE conference-ready figures and plots
- **🔬 State-of-the-Art Methodology**: Novel approach combining cost-sensitive learning with threshold optimization
- **📈 Advanced Metrics**: G-mean, balanced accuracy, and other imbalanced data metrics
- **🔄 Reproducible Research**: Complete setup and documentation for reproducibility
- **⚡ Efficient Processing**: Optimized for large-scale energy system data

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/NabidAlam/heat-pump-stress-detection.git
cd heat-pump-stress-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the analysis
```bash
python scripts/run_analysis.py --country None
```

## Results

Results will be generated in the `results/` directory:
- `pan_european_results.json` - Complete analysis results
- `pan_european_model.joblib` - Trained model
- `pan_european_*.png` - Generated plots

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list