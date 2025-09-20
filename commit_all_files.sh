#!/bin/bash

# Individual file commits for heat-pump-stress-detection project

echo "Starting individual file commits..."

# Configuration files
git add config/config.yaml
git commit -m "Add main configuration file with cost-sensitive learning settings"

git add config/config_fast.yaml
git commit -m "Add fast configuration for development iterations"

git add config/countries.yaml
git commit -m "Add pan-European countries configuration"

# Source Code - Data Modules
git add src/data/preprocessor.py
git commit -m "Add core data preprocessing module"

git add src/data/improved_preprocessor.py
git commit -m "Add advanced preprocessing with KNN imputation and normalization"

git add src/data/features.py
git commit -m "Add feature engineering module for heat pump data"

git add src/data/validation.py
git commit -m "Add data validation and cross-validation modules"

git add src/data/class_balancer.py
git commit -m "Add class balancing module for imbalanced datasets"

git add src/data/advanced_class_balancer.py
git commit -m "Add advanced ensemble class balancing strategies"

git add src/data/realistic_validator.py
git commit -m "Add realistic results validation for research quality"

git add src/data/loader.py
git commit -m "Add data loading utilities"

git add src/data/pipeline.py
git commit -m "Add data processing pipeline orchestration"

# Source Code - Model Modules
git add src/models/cost_sensitive_learner.py
git commit -m "Add cost-sensitive learning implementation for imbalanced data"

git add src/models/threshold_optimizer.py
git commit -m "Add threshold optimization for dynamic classification"

git add src/models/hyperparameter_tuner.py
git commit -m "Add hyperparameter tuning with Optuna optimization"

git add src/models/stress_detector.py
git commit -m "Add stress detection classifier implementation"

git add src/models/classifiers.py
git commit -m "Add machine learning classifier implementations"

git add src/models/calibration.py
git commit -m "Add model calibration for probability estimation"

git add src/models/monitoring.py
git commit -m "Add model training and evaluation monitoring"

# Source Code - DR and Evaluation
git add src/dr/heuristic.py
git commit -m "Add demand response heuristic algorithms"

git add src/dr/simulation.py
git commit -m "Add demand response simulation framework"

git add src/evaluation/metrics.py
git commit -m "Add comprehensive evaluation metrics for imbalanced data"

git add src/evaluation/validation.py
git commit -m "Add model validation and performance assessment"

git add src/evaluation/visualization.py
git commit -m "Add visualization utilities for results analysis"

# Scripts
git add scripts/run_analysis.py
git commit -m "Add main analysis script with cost-sensitive learning pipeline"

git add scripts/generate_figures.py
git commit -m "Add professional figure generation with comprehensive table generation for IEEE publications"

git add scripts/run_notebook_cells.py
git commit -m "Add EDA notebook execution script for data exploration"

# EDA Notebooks
git add notebooks/EDA_Notebooks_Guide.md
git commit -m "Add comprehensive EDA notebooks guide for data exploration"

git add notebooks/01_Data_Overview.ipynb
git commit -m "Add data overview and basic statistics EDA notebook"

git add notebooks/02_Heat_Demand_Analysis.ipynb
git commit -m "Add heat demand patterns and seasonality analysis notebook"

git add notebooks/03_COP_Efficiency_Analysis.ipynb
git commit -m "Add COP efficiency analysis and performance patterns notebook"

git add notebooks/04_Stress_Detection_Analysis.ipynb
git commit -m "Add stress detection patterns and threshold analysis notebook"

git add notebooks/05_Feature_Engineering_Analysis.ipynb
git commit -m "Add feature engineering and correlation analysis notebook"

git add notebooks/06_Model_Performance_Analysis.ipynb
git commit -m "Add model performance and evaluation analysis notebook"

git add notebooks/07_DR_Impact_Analysis.ipynb
git commit -m "Add demand response impact and intervention analysis notebook"

# Tests
git add tests/test_data_loader.py
git commit -m "Add data loading unit tests"

git add tests/test_cross_validation.py
git commit -m "Add cross-validation testing suite"

git add tests/test_evaluation_metrics.py
git commit -m "Add evaluation metrics testing"

git add tests/test_stress_detector.py
git commit -m "Add stress detection model tests"

git add tests/test_dr_heuristic.py
git commit -m "Add demand response heuristic tests"

git add tests/test_integration.py
git commit -m "Add integration testing for complete pipeline"

git add tests/test_model_persistence.py
git commit -m "Add model persistence and loading tests"

git add tests/conftest.py
git commit -m "Add pytest configuration and fixtures"

git add tests/README.md
git commit -m "Add test suite documentation"

# Results - Analysis Data
git add results/improved_conference_paper_results.json
git commit -m "Add complete analysis results with cost-sensitive learning metrics and threshold optimization"

# Results - State-of-the-Art Figures
git add results/sota_figures/figure_1_sota_timeline.png
git commit -m "Add state-of-the-art timeline visualization with pan-European data"

git add results/sota_figures/figure_1_sota_timeline.pdf
git commit -m "Add state-of-the-art timeline visualization PDF for publication"

git add results/sota_figures/figure_2_sota_performance.png
git commit -m "Add state-of-the-art performance analysis with radar charts and heatmaps"

git add results/sota_figures/figure_2_sota_performance.pdf
git commit -m "Add state-of-the-art performance analysis PDF for publication"

git add results/sota_figures/figure_3_sota_dr_analysis.png
git commit -m "Add state-of-the-art demand response analysis visualization"

git add results/sota_figures/figure_3_sota_dr_analysis.pdf
git commit -m "Add state-of-the-art demand response analysis PDF for publication"

# Results - Publication Tables
git add results/tables/table_1_model_performance.csv
git commit -m "Add model performance comparison table for paper writing"

git add results/tables/table_2_dr_analysis.csv
git commit -m "Add demand response analysis table for paper writing"

git add results/tables/table_3_dataset_characteristics.csv
git commit -m "Add dataset characteristics table for paper writing"

git add results/tables/table_4_threshold_optimization.csv
git commit -m "Add threshold optimization results table for paper writing"

git add results/tables/table_5_feature_importance.csv
git commit -m "Add feature importance analysis table for paper writing"

git add results/tables/table_6_statistical_significance.csv
git commit -m "Add statistical significance tests table for paper writing"

git add results/tables/table_summary.csv
git commit -m "Add table summary overview for paper writing"

# Structure Files
git add data/raw/.gitkeep
git commit -m "Add raw data directory structure preservation"

git add data/processed/.gitkeep
git commit -m "Add processed data directory structure preservation"

git add results/figures/.gitkeep
git commit -m "Add figures directory structure preservation"

git add results/monitoring/.gitkeep
git commit -m "Add monitoring directory structure preservation"

git add results/sota_figures/.gitkeep
git commit -m "Add state-of-the-art figures directory structure preservation"

git add results/tables/.gitkeep
git commit -m "Add tables directory structure preservation"

# Project Documentation
git add README.md
git commit -m "Add comprehensive project documentation with setup instructions"

git add requirements.txt
git commit -m "Add Python dependencies for reproducible environment"

git add setup.py
git commit -m "Add project setup script for easy installation"

git add .gitignore
git commit -m "Add Git ignore file for clean repository management"

git add .gitattributes
git commit -m "Add Git attributes for proper file handling"

echo "All individual file commits completed!"
echo "Total commits created: $(git rev-list --count HEAD)"
