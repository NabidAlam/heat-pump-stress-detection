# Heat Pump Stress Detection

A comprehensive machine learning project for detecting stress hours in heat pump systems using pan-European energy data from the When2Heat dataset. This project implements advanced cost-sensitive learning, threshold optimization, and demand response analysis for heat pump stress detection.

## Overview

This project addresses the critical challenge of identifying stress conditions in heat pump systems, which can lead to reduced efficiency and potential system failures. Using machine learning techniques and real-world energy data, we develop models to predict stress hours and analyze demand response strategies.

### Key Features

- **Advanced ML Pipeline**: Cost-sensitive learning with threshold optimization
- **Comprehensive Evaluation**: Multiple metrics for imbalanced classification
- **Demand Response Analysis**: Simulation and impact assessment
- **Publication-Ready Results**: IEEE-standard figures and tables
- **Reproducible Research**: Complete documentation and setup guides

## Project Structure

```
heat-pump-stress-detection/
├── src/                          # Core source code modules
│   ├── data/                     # Data processing and feature engineering
│   ├── models/                   # ML models and optimization
│   ├── evaluation/               # Metrics and validation
│   └── dr/                       # Demand response simulation
├── scripts/                      # Main analysis and figure generation
├── notebooks/                    # EDA notebooks for data exploration
├── config/                       # Configuration files
├── results/                      # Analysis results, figures, and tables
├── tests/                        # Unit tests and integration tests
└── data/                         # Data storage (raw and processed)
```

## Installation

### Option 1: Conda Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/NabidAlam/heat-pump-stress-detection.git
cd heat-pump-stress-detection

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate heat-pump-stress-detection
```

### Option 2: Pip Installation
```bash
# Clone the repository
git clone https://github.com/NabidAlam/heat-pump-stress-detection.git
cd heat-pump-stress-detection

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Automated Setup
```bash
# Run the automated setup script
python setup_environment.py
```

## Quick Start

### 1. Run the Complete Analysis
```bash
# Run the main analysis pipeline
python scripts/run_analysis.py

# Generate publication-ready figures and tables
python scripts/generate_figures.py
```

### 2. Explore the Data
```bash
# Run EDA notebooks
python scripts/run_notebook_cells.py
```

### 3. View Results
- **Analysis Results**: `results/improved_conference_paper_results.json`
- **Figures**: `results/sota_figures/`
- **Tables**: `results/tables/`

## Usage

### Main Analysis Script
```bash
python scripts/run_analysis.py
```

This script performs:
- Data loading and preprocessing
- Feature engineering
- Model training (Logistic Regression, Gradient Boosting, XGBoost)
- Cost-sensitive learning and threshold optimization
- Rolling-origin validation
- Demand response simulation

### Figure Generation
```bash
python scripts/generate_figures.py
```


### EDA Notebooks
```bash
python scripts/run_notebook_cells.py
```

Runs comprehensive exploratory data analysis:
- Data overview and statistics
- Heat demand patterns
- COP efficiency analysis
- Stress detection patterns
- Feature engineering analysis
- Model performance analysis
- DR impact analysis

## Configuration

The project uses YAML configuration files in the `config/` directory:

- `config.yaml`: Main configuration with all parameters
- `config_focused.yaml`: Focused analysis configuration

Key configuration sections:
- Data paths and parameters
- Stress detection thresholds
- Model hyperparameters
- Evaluation settings
- Cost-sensitive learning parameters

## Results

### Model Performance
- **XGBoost**: F1-Score: 0.765, AUC-ROC: 0.993
- **Gradient Boosting**: F1-Score: 0.711, AUC-ROC: 0.988
- **Logistic Regression**: F1-Score: 0.605, AUC-ROC: 0.954

### Key Findings
- Heat demand patterns show strong seasonal variations
- COP efficiency correlates with stress detection
- Cost-sensitive learning significantly improves minority class performance
- Demand response strategies show measurable impact on peak reduction

## Dataset

The project uses the **When2Heat** dataset from the Open Power System Data platform:
- **Coverage**: Pan-European (28 countries)
- **Time Period**: 2008-2022 (hourly resolution)
- **Features**: Heat demand, COP efficiency, temporal features
- **Stress Rate**: ~1.5% (highly imbalanced)
- **Data Source**: [When2Heat Heating Profiles](https://data.open-power-system-data.org/when2heat/2023-07-27)

### Dataset Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{when2heat_2023,
  title={When2Heat Heating Profiles},
  author={Ruhnau, Oliver and Muessel, Jarusch},
  year={2023},
  publisher={Open Power System Data},
  doi={10.25832/when2heat/2023-07-27},
  url={https://data.open-power-system-data.org/when2heat/2023-07-27}
}
```

**Original Dataset Citation:**
```bibtex
@article{ruhnau2019time,
  title={Time series of heat demand and heat pump efficiency for energy system modeling},
  author={Ruhnau, Oliver and Hirth, Lion and Praktiknjo, Aaron},
  journal={Scientific Data},
  volume={6},
  number={1},
  pages={189},
  year={2019},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-019-0199-y}
}
```

**Working Paper Citation:**
```bibtex
@techreport{ruhnau2023update,
  title={Update and extension of the When2Heat dataset},
  author={Ruhnau, Oliver and Muessel, Jarusch},
  year={2023},
  institution={Econstor},
  url={http://hdl.handle.net/10419/249997}
}
```

## Technical Details

### Machine Learning Approach
- **Problem Type**: Binary classification (stress vs. normal)
- **Challenge**: Highly imbalanced dataset
- **Solution**: Cost-sensitive learning + threshold optimization
- **Validation**: Rolling-origin temporal validation
- **Metrics**: F1-score, G-mean, AUC-ROC, AUC-PR

### Advanced Features
- **Feature Engineering**: Rolling statistics, lag features, seasonal indicators
- **Preprocessing**: KNN imputation, outlier detection, robust scaling
- **Model Selection**: Multiple algorithms with hyperparameter tuning
- **Threshold Optimization**: F1-score, G-mean, balanced accuracy optimization

## Documentation

- **Setup Guide**: `SETUP_GUIDE.md` - Detailed installation instructions
- **EDA Guide**: `notebooks/EDA_Notebooks_Guide.md` - Data exploration guide
- **Configuration**: `config/` - Parameter documentation

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{heat_pump_stress_detection,
  title={Heat Pump Stress Detection: A Machine Learning Approach},
  author={Md Shahabub Alam},
  year={2025},
  url={https://github.com/NabidAlam/heat-pump-stress-detection}
}
```
