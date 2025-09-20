#!/usr/bin/env python3
"""
Improved Data Preprocessor for Energy System Analysis
====================================================

This module provides advanced preprocessing capabilities including:
- KNN imputation for missing values
- Robust feature scaling
- Outlier detection and removal
- Time-based feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from loguru import logger
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


class ImprovedDataPreprocessor:
    """Advanced data preprocessor for energy system data."""
    
    def __init__(self, 
                 imputation_method: str = 'knn',
                 scaling_method: str = 'robust',
                 outlier_detection: bool = True,
                 outlier_contamination: float = 0.1,
                 random_state: int = 42):
        """
        Initialize the improved preprocessor.
        
        Args:
            imputation_method: Method for missing value imputation ('knn', 'mean', 'median')
            scaling_method: Method for feature scaling ('robust', 'standard', 'minmax')
            outlier_detection: Whether to detect and remove outliers
            outlier_contamination: Expected proportion of outliers
            random_state: Random state for reproducibility
        """
        self.imputation_method = imputation_method
        self.scaling_method = scaling_method
        self.outlier_detection = outlier_detection
        self.outlier_contamination = outlier_contamination
        self.random_state = random_state
        
        # Initialize components
        self.imputer = None
        self.scaler = None
        self.outlier_detector = None
        self.feature_columns = None
        self.is_fitted = False
        
        logger.info(f"Initialized ImprovedDataPreprocessor with {imputation_method} imputation and {scaling_method} scaling")
    
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
        
        logger.info("Created time-based features")
        return data
    
    def create_aggregated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated statistical features."""
        data = data.copy()
        
        # Rolling statistics for key variables
        if 'heat_demand_total' in data.columns:
            data['demand_rolling_mean_24h'] = data['heat_demand_total'].rolling(window=24, center=False).mean()
            data['demand_rolling_std_24h'] = data['heat_demand_total'].rolling(window=24, center=False).std()
            data['demand_rolling_max_24h'] = data['heat_demand_total'].rolling(window=24, center=False).max()
        
        if 'cop_average' in data.columns:
            data['cop_rolling_mean_24h'] = data['cop_average'].rolling(window=24, center=False).mean()
            data['cop_rolling_min_24h'] = data['cop_average'].rolling(window=24, center=False).min()
        
        # Lag features (avoiding future data leakage)
        if 'heat_demand_total' in data.columns:
            data['demand_lag_1h'] = data['heat_demand_total'].shift(1)
            data['demand_lag_24h'] = data['heat_demand_total'].shift(24)
        
        if 'cop_average' in data.columns:
            data['cop_lag_1h'] = data['cop_average'].shift(1)
            data['cop_lag_24h'] = data['cop_average'].shift(24)
        
        logger.info("Created aggregated statistical features")
        return data
    
    def detect_stress_hours_dynamic(self, data: pd.DataFrame, 
                                  demand_percentile: float = 90,
                                  cop_percentile: float = 10,
                                  seasonal_window: int = 168) -> pd.DataFrame:
        """
        Detect stress hours using dynamic seasonal thresholds.
        
        Args:
            data: Input dataframe
            demand_percentile: Percentile for high demand threshold
            cop_percentile: Percentile for low COP threshold
            seasonal_window: Window size for seasonal threshold calculation
        """
        data = data.copy()
        
        if 'heat_demand_total' not in data.columns or 'cop_average' not in data.columns:
            logger.warning("Required columns not found for stress detection")
            data['stress_hour'] = 0
            return data
        
        # Calculate dynamic thresholds using rolling percentiles
        data['demand_threshold'] = data['heat_demand_total'].rolling(
            window=seasonal_window, center=False
        ).quantile(demand_percentile / 100)
        
        data['cop_threshold'] = data['cop_average'].rolling(
            window=seasonal_window, center=False
        ).quantile(cop_percentile / 100)
        
        # Detect stress conditions
        high_demand = data['heat_demand_total'] >= data['demand_threshold']
        low_cop = data['cop_average'] <= data['cop_threshold']
        
        # Stress hour is when both conditions are met
        data['stress_hour'] = (high_demand & low_cop).astype(int)
        data['high_demand'] = high_demand.astype(int)
        data['low_cop'] = low_cop.astype(int)
        
        stress_rate = data['stress_hour'].mean()
        logger.info(f"Detected stress hours with {stress_rate:.3f} stress rate")
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values using specified imputation method."""
        data = data.copy()
        
        # Select numeric columns for imputation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        missing_cols = data[numeric_cols].columns[data[numeric_cols].isnull().any()].tolist()
        
        if not missing_cols:
            logger.info("No missing values found")
            return data
        
        logger.info(f"Found missing values in {len(missing_cols)} columns: {missing_cols}")
        
        if self.imputation_method == 'knn':
            if self.imputer is None or not self.is_fitted:
                self.imputer = KNNImputer(n_neighbors=5)
            
            if fit:
                data[numeric_cols] = self.imputer.fit_transform(data[numeric_cols])
                self.is_fitted = True
            else:
                if not self.is_fitted:
                    raise ValueError("Imputer must be fitted before transforming new data")
                data[numeric_cols] = self.imputer.transform(data[numeric_cols])
        
        elif self.imputation_method == 'mean':
            if fit:
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
            else:
                # Use pre-computed means for new data
                pass
        
        elif self.imputation_method == 'median':
            if fit:
                data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
            else:
                # Use pre-computed medians for new data
                pass
        
        logger.info(f"Imputed missing values using {self.imputation_method} method")
        return data
    
    def detect_outliers(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Detect and optionally remove outliers."""
        if not self.outlier_detection:
            return data
        
        data = data.copy()
        
        # Select numeric columns for outlier detection
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['stress_hour', 'high_demand', 'low_cop']]
        
        if not feature_cols:
            logger.warning("No feature columns found for outlier detection")
            return data
        
        if self.outlier_detector is None or not self.is_fitted:
            self.outlier_detector = IsolationForest(
                contamination=self.outlier_contamination,
                random_state=self.random_state
            )
        
        if fit:
            outlier_labels = self.outlier_detector.fit_predict(data[feature_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Outlier detector must be fitted before transforming new data")
            outlier_labels = self.outlier_detector.predict(data[feature_cols])
        
        # Remove outliers (label -1 indicates outliers)
        outlier_mask = outlier_labels == -1
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            logger.info(f"Detected {n_outliers} outliers ({n_outliers/len(data)*100:.2f}% of data)")
            data = data[~outlier_mask].copy()
            logger.info(f"Removed outliers, remaining data: {len(data)} rows")
        
        return data
    
    def normalize_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize features using specified scaling method."""
        data = data.copy()
        
        # Select numeric columns (exclude target, metadata, and core physical variables)
        exclude_cols = ['stress_hour', 'high_demand', 'low_cop', 'demand_threshold', 'cop_threshold',
                       'hour', 'day_of_week', 'day_of_year', 'month', 'year', 'is_weekend', 'is_winter', 'is_summer',
                       'heat_demand_total', 'cop_average', 'cop_ashp']  # CRITICAL: Don't normalize core physical variables
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if not feature_cols:
            logger.warning("No numeric feature columns found for normalization")
            return data
        
        # Initialize scaler if not already done
        if self.scaler is None:
            if self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            elif self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.scaling_method}. Using RobustScaler.")
                self.scaler = RobustScaler()
        
        # Fit and transform
        if fit:
            data[feature_cols] = self.scaler.fit_transform(data[feature_cols])
            self.feature_columns = feature_cols
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before transforming new data")
            data[feature_cols] = self.scaler.transform(data[feature_cols])
        
        logger.info(f"Normalized {len(feature_cols)} feature columns")
        
        return data
    
    def preprocess(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Input dataframe
            fit: Whether to fit the preprocessors (True for training data)
        
        Returns:
            Preprocessed dataframe
        """
        logger.info(f"Starting preprocessing pipeline (fit={fit})")
        
        # Step 1: Create time features
        data = self.create_time_features(data)
        
        # Step 2: Create aggregated features
        data = self.create_aggregated_features(data)
        
        # Step 3: Detect stress hours
        data = self.detect_stress_hours_dynamic(data)
        
        # Step 4: Handle missing values
        data = self.handle_missing_values(data, fit=fit)
        
        # Step 5: Detect outliers
        data = self.detect_outliers(data, fit=fit)
        
        # Step 6: Normalize features
        data = self.normalize_features(data, fit=fit)
        
        logger.info(f"Preprocessing completed. Final data shape: {data.shape}")
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        if self.feature_columns is None:
            logger.warning("Feature columns not set. Run preprocessing first.")
            return []
        return self.feature_columns.copy()
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing configuration."""
        return {
            'imputation_method': self.imputation_method,
            'scaling_method': self.scaling_method,
            'outlier_detection': self.outlier_detection,
            'outlier_contamination': self.outlier_contamination,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns
        }
