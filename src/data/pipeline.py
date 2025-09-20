#!/usr/bin/env python3
"""
Data Processing Pipeline
========================

This module provides a comprehensive data processing pipeline for
energy system stress detection, integrating preprocessing, feature
engineering, and validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from .improved_preprocessor import ImprovedDataPreprocessor
from .features import FeatureEngineer


class DataPipeline:
    """
    Comprehensive data processing pipeline for energy system analysis.
    
    Integrates data loading, preprocessing, feature engineering, and validation
    into a single pipeline.
    """
    
    def __init__(self, 
                 config: Dict[str, Any],
                 random_state: int = 42):
        """
        Initialize the data pipeline.
        
        Args:
            config: Configuration dictionary
            random_state: Random state for reproducibility
        """
        self.config = config
        self.random_state = random_state
        
        # Initialize components
        self.preprocessor = ImprovedDataPreprocessor(
            imputation_method=config.get('preprocessing', {}).get('imputation_method', 'knn'),
            scaling_method=config.get('preprocessing', {}).get('scaling_method', 'robust'),
            outlier_detection=config.get('preprocessing', {}).get('outlier_detection', True),
            random_state=random_state
        )
        
        self.feature_engineer = FeatureEngineer(
            rolling_windows=config.get('features', {}).get('rolling_windows', [24, 168]),
            lag_periods=config.get('features', {}).get('lag_periods', [1, 24]),
            random_state=random_state
        )
        
        logger.info("Initialized DataPipeline")
    
    def load_data(self, data_path: str, 
                 country_code: Optional[str] = None,
                 years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load and prepare data for processing.
        
        Args:
            data_path: Path to the data file
            country_code: Country code for filtering (None for pan-European)
            years: Years to include (None for all years)
        
        Returns:
            Loaded and prepared dataframe
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        
        # Filter by years if specified
        if years is not None:
            data = data[data.index.year.isin(years)]
            logger.info(f"Filtered data to years: {years}")
        
        # Handle country-specific data
        if country_code is not None and country_code != 'pan_european':
            if country_code in data.columns:
                data = data[data[country_code].notna()].copy()
                data = data.rename(columns={country_code: 'heat_demand_total'})
                logger.info(f"Loaded data for country: {country_code}")
            else:
                logger.warning(f"Country {country_code} not found, using pan-European data")
                # Use pan-European aggregation
                eu_countries = [col for col in data.columns if col in [
                    'AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
                    'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV',
                    'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'
                ]]
                data['heat_demand_total'] = data[eu_countries].sum(axis=1)
                logger.info(f"Created pan-European heat demand from {len(eu_countries)} countries")
        else:
            # Pan-European analysis
            eu_countries = [col for col in data.columns if col in [
                'AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
                'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV',
                'NL', 'NO', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'
            ]]
            data['heat_demand_total'] = data[eu_countries].sum(axis=1)
            logger.info(f"Created pan-European heat demand from {len(eu_countries)} countries")
        
        # Add synthetic COP data (in real implementation, this would come from the dataset)
        np.random.seed(self.random_state)
        
        # Create realistic COP variation
        base_cop = 3.5
        demand_factor = (data['heat_demand_total'] - data['heat_demand_total'].mean()) / data['heat_demand_total'].std()
        seasonal_factor = np.sin(2 * np.pi * data.index.dayofyear / 365) * 0.3
        
        data['cop_average'] = base_cop + demand_factor * 0.2 + seasonal_factor + np.random.normal(0, 0.1, len(data))
        data['cop_average'] = np.clip(data['cop_average'], 2.0, 5.0)
        
        data['cop_ashp'] = data['cop_average'] * np.random.uniform(0.9, 1.1, len(data))
        data['cop_ashp'] = np.clip(data['cop_ashp'], 2.0, 5.0)
        
        logger.info(f"Loaded {len(data)} data points")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        logger.info(f"Heat demand range: {data['heat_demand_total'].min():.1f} - {data['heat_demand_total'].max():.1f} MW")
        
        return data
    
    def process_data(self, data: pd.DataFrame, 
                    fit_preprocessing: bool = True) -> pd.DataFrame:
        """
        Process data through the complete pipeline.
        
        Args:
            data: Input dataframe
            fit_preprocessing: Whether to fit preprocessing components
        
        Returns:
            Processed dataframe
        """
        logger.info("Starting data processing pipeline")
        
        # Step 1: Preprocessing
        processed_data = self.preprocessor.preprocess(data, fit=fit_preprocessing)
        
        # Step 2: Feature engineering
        target_columns = ['heat_demand_total', 'cop_average']
        processed_data = self.feature_engineer.engineer_features(
            processed_data, target_columns
        )
        
        # Step 3: Validation
        self._validate_data(processed_data)
        
        logger.info(f"Data processing completed. Final shape: {processed_data.shape}")
        
        return processed_data
    
    def _validate_data(self, data: pd.DataFrame):
        """Validate processed data."""
        logger.info("Validating processed data")
        
        # Check for required columns
        required_columns = ['heat_demand_total', 'cop_average', 'stress_hour']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataset
        if len(data) == 0:
            raise ValueError("Processed dataset is empty")
        
        # Check stress rate
        stress_rate = data['stress_hour'].mean()
        if stress_rate < 0.001:
            logger.warning(f"Very low stress rate: {stress_rate:.4f}")
        elif stress_rate > 0.5:
            logger.warning(f"Very high stress rate: {stress_rate:.4f}")
        
        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"Found infinite values in columns: {inf_cols}")
        
        logger.info("Data validation completed")
    
    def get_feature_matrix(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Get feature matrix and feature names for machine learning."""
        feature_matrix = self.feature_engineer.create_feature_matrix(data)
        feature_names = self.feature_engineer.feature_names
        
        return feature_matrix.values, feature_names
    
    def get_target_vector(self, data: pd.DataFrame) -> np.ndarray:
        """Get target vector for machine learning."""
        target = self.feature_engineer.get_target_vector(data)
        return target.values
    
    def save_processed_data(self, data: pd.DataFrame, 
                          output_path: str,
                          country_code: Optional[str] = None):
        """Save processed data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        if country_code is not None:
            filename = f"{country_code}_processed_data.csv"
        else:
            filename = "pan_european_processed_data.csv"
        
        file_path = output_path.parent / filename
        
        # Save data
        data.to_csv(file_path)
        logger.info(f"Processed data saved to {file_path}")
        
        # Save feature information
        feature_info = self.feature_engineer.get_feature_importance_info()
        feature_info_path = output_path.parent / f"{filename.replace('.csv', '_features.json')}"
        
        import json
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info(f"Feature information saved to {feature_info_path}")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        return {
            'preprocessing': self.preprocessor.get_preprocessing_info(),
            'feature_engineering': self.feature_engineer.get_feature_importance_info(),
            'config': self.config,
            'random_state': self.random_state
        }
