#!/usr/bin/env python3
"""
Data Loading Module
==================

This module provides data loading utilities for the energy system analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Data loader for energy system datasets.
    
    Provides utilities for loading and validating energy system data
    from various sources and formats.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataLoader with data directory: {self.data_dir}")
    
    def load_when2heat_data(self, 
                          country_codes: Optional[List[str]] = None,
                          years: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Load When2Heat dataset.
        
        Args:
            country_codes: List of country codes to load (None for all)
            years: List of years to load (None for all)
        
        Returns:
            Loaded dataframe
        """
        data_path = self.data_dir / 'when2heat.csv'
        
        if not data_path.exists():
            raise FileNotFoundError(f"When2Heat data not found at {data_path}")
        
        logger.info(f"Loading When2Heat data from {data_path}")
        
        # Load data
        data = pd.read_csv(data_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
        
        # Filter by years if specified
        if years is not None:
            data = data[data.index.year.isin(years)]
            logger.info(f"Filtered data to years: {years}")
        
        # Filter by countries if specified
        if country_codes is not None:
            available_countries = [col for col in country_codes if col in data.columns]
            if available_countries:
                data = data[available_countries]
                logger.info(f"Filtered data to countries: {available_countries}")
            else:
                logger.warning(f"None of the specified countries found in data")
        
        logger.info(f"Loaded {len(data)} data points")
        logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data.
        
        Args:
            data: Dataframe to validate
        
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for empty dataset
        if len(data) == 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Dataset is empty")
            return validation_results
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            validation_results['warnings'].append("Index is not datetime")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        high_missing = missing_counts[missing_counts > len(data) * 0.1]
        if len(high_missing) > 0:
            validation_results['warnings'].append(f"High missing values in columns: {high_missing.index.tolist()}")
        
        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            validation_results['warnings'].append(f"Infinite values in columns: {inf_cols}")
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'date_range': {
                'start': data.index.min().isoformat() if isinstance(data.index, pd.DatetimeIndex) else None,
                'end': data.index.max().isoformat() if isinstance(data.index, pd.DatetimeIndex) else None
            },
            'missing_values': missing_counts.to_dict(),
            'numeric_columns': numeric_cols.tolist()
        }
        
        logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
        if validation_results['issues']:
            logger.error(f"Validation issues: {validation_results['issues']}")
        if validation_results['warnings']:
            logger.warning(f"Validation warnings: {validation_results['warnings']}")
        
        return validation_results
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            data: Dataframe to analyze
        
        Returns:
            Dataset information
        """
        info = {
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'index_info': {
                'type': str(type(data.index)),
                'is_datetime': isinstance(data.index, pd.DatetimeIndex)
            }
        }
        
        if isinstance(data.index, pd.DatetimeIndex):
            info['index_info'].update({
                'start': data.index.min().isoformat(),
                'end': data.index.max().isoformat(),
                'frequency': str(data.index.freq) if data.index.freq else 'irregular'
            })
        
        # Numeric column statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_statistics'] = data[numeric_cols].describe().to_dict()
        
        return info


def create_data_loader(data_dir: str = 'data/raw') -> DataLoader:
    """
    Create a data loader instance.
    
    Args:
        data_dir: Directory containing raw data files
    
    Returns:
        DataLoader instance
    """
    return DataLoader(data_dir)
