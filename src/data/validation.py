#!/usr/bin/env python3
"""
Data Validation Module
=====================

This module provides data validation utilities for ensuring data quality
and integrity throughout the analysis pipeline.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """
    Data validator for energy system datasets.
    
    Provides comprehensive validation checks for data quality,
    consistency, and integrity.
    """
    
    def __init__(self, 
                 min_stress_rate: float = 0.001,
                 max_stress_rate: float = 0.5,
                 max_missing_ratio: float = 0.1):
        """
        Initialize the data validator.
        
        Args:
            min_stress_rate: Minimum acceptable stress rate
            max_stress_rate: Maximum acceptable stress rate
            max_missing_ratio: Maximum acceptable missing value ratio
        """
        self.min_stress_rate = min_stress_rate
        self.max_stress_rate = max_stress_rate
        self.max_missing_ratio = max_missing_ratio
        
        logger.info("Initialized DataValidator")
    
    def validate_raw_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate raw data before preprocessing.
        
        Args:
            data: Raw dataframe to validate
        
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Basic checks
        if len(data) == 0:
            results['is_valid'] = False
            results['issues'].append("Dataset is empty")
            return results
        
        # Check required columns
        required_columns = ['heat_demand_total', 'cop_average']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            results['is_valid'] = False
            results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'heat_demand_total' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['heat_demand_total']):
                results['issues'].append("heat_demand_total must be numeric")
        
        if 'cop_average' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['cop_average']):
                results['issues'].append("cop_average must be numeric")
        
        # Check for reasonable value ranges
        if 'heat_demand_total' in data.columns:
            if (data['heat_demand_total'] < 0).any():
                results['warnings'].append("Negative heat demand values found")
            
            if (data['heat_demand_total'] > 1e6).any():
                results['warnings'].append("Unusually high heat demand values found")
        
        if 'cop_average' in data.columns:
            if (data['cop_average'] < 1.0).any():
                results['warnings'].append("COP values below 1.0 found (unrealistic)")
            
            if (data['cop_average'] > 10.0).any():
                results['warnings'].append("Unusually high COP values found")
        
        # Check for missing values
        missing_ratio = data.isnull().sum() / len(data)
        high_missing = missing_ratio[missing_ratio > self.max_missing_ratio]
        if len(high_missing) > 0:
            results['warnings'].append(f"High missing value ratio in columns: {high_missing.index.tolist()}")
        
        # Calculate statistics
        results['statistics'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_ratios': missing_ratio.to_dict(),
            'numeric_columns': data.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        if 'heat_demand_total' in data.columns:
            results['statistics']['heat_demand'] = {
                'min': float(data['heat_demand_total'].min()),
                'max': float(data['heat_demand_total'].max()),
                'mean': float(data['heat_demand_total'].mean()),
                'std': float(data['heat_demand_total'].std())
            }
        
        if 'cop_average' in data.columns:
            results['statistics']['cop_average'] = {
                'min': float(data['cop_average'].min()),
                'max': float(data['cop_average'].max()),
                'mean': float(data['cop_average'].mean()),
                'std': float(data['cop_average'].std())
            }
        
        logger.info(f"Raw data validation completed. Valid: {results['is_valid']}")
        return results
    
    def validate_processed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate processed data after preprocessing.
        
        Args:
            data: Processed dataframe to validate
        
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for stress hour column
        if 'stress_hour' not in data.columns:
            results['is_valid'] = False
            results['issues'].append("Missing stress_hour column")
            return results
        
        # Check stress rate
        stress_rate = data['stress_hour'].mean()
        if stress_rate < self.min_stress_rate:
            results['warnings'].append(f"Very low stress rate: {stress_rate:.4f}")
        elif stress_rate > self.max_stress_rate:
            results['warnings'].append(f"Very high stress rate: {stress_rate:.4f}")
        
        # Check for infinite values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            results['issues'].append(f"Infinite values in columns: {inf_cols}")
        
        # Check for NaN values in critical columns
        critical_columns = ['stress_hour', 'heat_demand_total', 'cop_average']
        nan_cols = []
        for col in critical_columns:
            if col in data.columns and data[col].isnull().any():
                nan_cols.append(col)
        
        if nan_cols:
            results['issues'].append(f"NaN values in critical columns: {nan_cols}")
        
        # Calculate statistics
        results['statistics'] = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'stress_rate': float(stress_rate),
            'stress_hours': int(data['stress_hour'].sum()),
            'feature_columns': len(numeric_cols) - 1  # Exclude target
        }
        
        logger.info(f"Processed data validation completed. Valid: {results['is_valid']}")
        return results
    
    def validate_model_inputs(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Validate model inputs.
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            Validation results
        """
        results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check shapes
        if len(X) != len(y):
            results['is_valid'] = False
            results['issues'].append(f"Feature matrix and target vector length mismatch: {len(X)} vs {len(y)}")
        
        # Check for empty arrays
        if len(X) == 0:
            results['is_valid'] = False
            results['issues'].append("Empty feature matrix")
        
        if len(y) == 0:
            results['is_valid'] = False
            results['issues'].append("Empty target vector")
        
        # Check for NaN values
        if np.isnan(X).any():
            results['issues'].append("NaN values in feature matrix")
        
        if np.isnan(y).any():
            results['issues'].append("NaN values in target vector")
        
        # Check for infinite values
        if np.isinf(X).any():
            results['issues'].append("Infinite values in feature matrix")
        
        if np.isinf(y).any():
            results['issues'].append("Infinite values in target vector")
        
        # Check target distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            results['warnings'].append("Only one class present in target")
        
        class_ratio = counts[1] / counts[0] if len(counts) > 1 else 0
        if class_ratio < 0.01:
            results['warnings'].append(f"Extreme class imbalance: {class_ratio:.4f}")
        
        # Calculate statistics
        results['statistics'] = {
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X) > 0 else 0,
            'n_classes': len(unique_classes),
            'class_distribution': dict(zip(unique_classes.astype(str), counts.astype(int))),
            'class_ratio': float(class_ratio)
        }
        
        logger.info(f"Model input validation completed. Valid: {results['is_valid']}")
        return results
    
    def validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate analysis results.
        
        Args:
            results: Analysis results to validate
        
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check for required result keys
        required_keys = ['metrics', 'model_info']
        missing_keys = [key for key in required_keys if key not in results]
        if missing_keys:
            validation_results['warnings'].append(f"Missing result keys: {missing_keys}")
        
        # Validate metrics
        if 'metrics' in results:
            metrics = results['metrics']
            required_metrics = ['accuracy', 'precision', 'recall', 'f1']
            missing_metrics = [metric for metric in required_metrics if metric not in metrics]
            if missing_metrics:
                validation_results['warnings'].append(f"Missing metrics: {missing_metrics}")
            
            # Check metric ranges
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if metric in metrics:
                    value = metrics[metric]
                    if not (0 <= value <= 1):
                        validation_results['warnings'].append(f"{metric} out of range [0,1]: {value}")
        
        logger.info(f"Results validation completed. Valid: {validation_results['is_valid']}")
        return validation_results


def create_data_validator(**kwargs) -> DataValidator:
    """
    Create a data validator instance.
    
    Args:
        **kwargs: Validator configuration parameters
    
    Returns:
        DataValidator instance
    """
    return DataValidator(**kwargs)
