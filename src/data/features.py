#!/usr/bin/env python3
"""
Feature Engineering Module
==========================

This module provides feature engineering capabilities for energy system
stress detection, including time-based features, rolling statistics,
and lag features.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for energy system stress detection.
    
    Creates time-based features, rolling statistics, and lag features
    for machine learning models.
    """
    
    def __init__(self, 
                 rolling_windows: List[int] = [24, 168],  # 24h and 1 week
                 lag_periods: List[int] = [1, 24],       # 1h and 1 day
                 random_state: int = 42):
        """
        Initialize the feature engineer.
        
        Args:
            rolling_windows: List of rolling window sizes in hours
            lag_periods: List of lag periods in hours
            random_state: Random state for reproducibility
        """
        self.rolling_windows = rolling_windows
        self.lag_periods = lag_periods
        self.random_state = random_state
        self.feature_names = []
        
        logger.info(f"Initialized FeatureEngineer with windows {rolling_windows} and lags {lag_periods}")
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime index."""
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data index is not datetime, skipping time feature creation")
            return data
        
        data = data.copy()
        
        # Basic time features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['day_of_year'] = data.index.dayofyear
        data['month'] = data.index.month
        data['year'] = data.index.year
        
        # Derived time features
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_winter'] = data['month'].isin([12, 1, 2]).astype(int)
        data['is_summer'] = data['month'].isin([6, 7, 8]).astype(int)
        data['is_spring'] = data['month'].isin([3, 4, 5]).astype(int)
        data['is_autumn'] = data['month'].isin([9, 10, 11]).astype(int)
        
        # Cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        logger.info("Created time-based features")
        return data
    
    def create_rolling_features(self, data: pd.DataFrame, 
                              target_columns: List[str]) -> pd.DataFrame:
        """Create rolling statistical features."""
        data = data.copy()
        
        for column in target_columns:
            if column not in data.columns:
                logger.warning(f"Column {column} not found, skipping rolling features")
                continue
            
            for window in self.rolling_windows:
                # Rolling statistics
                data[f'{column}_rolling_mean_{window}h'] = data[column].rolling(
                    window=window, center=False
                ).mean()
                
                data[f'{column}_rolling_std_{window}h'] = data[column].rolling(
                    window=window, center=False
                ).std()
                
                data[f'{column}_rolling_max_{window}h'] = data[column].rolling(
                    window=window, center=False
                ).max()
                
                data[f'{column}_rolling_min_{window}h'] = data[column].rolling(
                    window=window, center=False
                ).min()
                
                # Rolling percentiles
                data[f'{column}_rolling_p90_{window}h'] = data[column].rolling(
                    window=window, center=False
                ).quantile(0.9)
                
                data[f'{column}_rolling_p10_{window}h'] = data[column].rolling(
                    window=window, center=False
                ).quantile(0.1)
        
        logger.info(f"Created rolling features for {len(target_columns)} columns")
        return data
    
    def create_lag_features(self, data: pd.DataFrame, 
                          target_columns: List[str]) -> pd.DataFrame:
        """Create lag features."""
        data = data.copy()
        
        for column in target_columns:
            if column not in data.columns:
                logger.warning(f"Column {column} not found, skipping lag features")
                continue
            
            for lag in self.lag_periods:
                data[f'{column}_lag_{lag}h'] = data[column].shift(lag)
        
        logger.info(f"Created lag features for {len(target_columns)} columns")
        return data
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        data = data.copy()
        
        # Demand-COP interactions
        if 'heat_demand_total' in data.columns and 'cop_average' in data.columns:
            data['demand_cop_ratio'] = data['heat_demand_total'] / data['cop_average']
            data['demand_cop_product'] = data['heat_demand_total'] * data['cop_average']
        
        # Seasonal demand patterns
        if 'heat_demand_total' in data.columns and 'is_winter' in data.columns:
            data['winter_demand'] = data['heat_demand_total'] * data['is_winter']
            data['summer_demand'] = data['heat_demand_total'] * data['is_summer']
        
        # Peak hour indicators
        if 'hour' in data.columns and 'heat_demand_total' in data.columns:
            data['is_peak_hour'] = data['hour'].isin([17, 18, 19, 20, 21]).astype(int)
            data['peak_demand'] = data['heat_demand_total'] * data['is_peak_hour']
        
        logger.info("Created interaction features")
        return data
    
    def create_feature_matrix(self, data: pd.DataFrame, 
                            feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Create feature matrix for machine learning."""
        if feature_columns is None:
            # Select numeric columns excluding target and metadata
            exclude_cols = ['stress_hour', 'high_demand', 'low_cop', 
                          'demand_threshold', 'cop_threshold']
            feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_cols]
        
        # Create feature matrix
        feature_matrix = data[feature_columns].copy()
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        self.feature_names = feature_columns
        logger.info(f"Created feature matrix with {len(feature_columns)} features")
        
        return feature_matrix
    
    def get_target_vector(self, data: pd.DataFrame, 
                         target_column: str = 'stress_hour') -> pd.Series:
        """Get target vector for machine learning."""
        if target_column not in data.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        
        target = data[target_column].copy()
        logger.info(f"Created target vector with {target.sum()} positive samples ({target.mean():.3f} rate)")
        
        return target
    
    def engineer_features(self, data: pd.DataFrame, 
                         target_columns: List[str] = None) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            data: Input dataframe
            target_columns: Columns to create rolling and lag features for
        
        Returns:
            Dataframe with engineered features
        """
        if target_columns is None:
            target_columns = ['heat_demand_total', 'cop_average']
        
        logger.info("Starting feature engineering pipeline")
        
        # Step 1: Time features
        data = self.create_time_features(data)
        
        # Step 2: Rolling features
        data = self.create_rolling_features(data, target_columns)
        
        # Step 3: Lag features
        data = self.create_lag_features(data, target_columns)
        
        # Step 4: Interaction features
        data = self.create_interaction_features(data)
        
        logger.info(f"Feature engineering completed. Final shape: {data.shape}")
        
        return data
    
    def get_feature_importance_info(self) -> Dict[str, Any]:
        """Get information about created features."""
        return {
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'rolling_windows': self.rolling_windows,
            'lag_periods': self.lag_periods,
            'feature_categories': {
                'time_features': ['hour', 'day_of_week', 'month', 'is_weekend', 'is_winter'],
                'rolling_features': [f'*_rolling_*_{w}h' for w in self.rolling_windows],
                'lag_features': [f'*_lag_{l}h' for l in self.lag_periods],
                'interaction_features': ['demand_cop_ratio', 'winter_demand', 'peak_demand']
            }
        }
