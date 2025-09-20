#!/usr/bin/env python3
"""
Test Data Loading Functionality
===============================

Tests for data loading and preprocessing functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.improved_preprocessor import ImprovedDataPreprocessor


class TestDataLoader(unittest.TestCase):
    """Test data loading and preprocessing functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        
        self.data = pd.DataFrame({
            'heat_demand_total': np.random.exponential(100, n_samples),
            'cop_average': np.random.uniform(2, 5, n_samples),
            'cop_ashp': np.random.uniform(2, 5, n_samples)
        })
        self.data.index = pd.date_range('2020-01-01', periods=n_samples, freq='h')
    
    def test_data_loading(self):
        """Test basic data loading functionality."""
        # Test that data has expected columns
        expected_columns = ['heat_demand_total', 'cop_average', 'cop_ashp']
        for col in expected_columns:
            self.assertIn(col, self.data.columns)
        
        # Test data types
        self.assertTrue(pd.api.types.is_numeric_dtype(self.data['heat_demand_total']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.data['cop_average']))
    
    def test_preprocessing_pipeline(self):
        """Test the complete preprocessing pipeline."""
        preprocessor = ImprovedDataPreprocessor(
            imputation_method='knn',
            scaling_method='robust',
            random_state=42
        )
        
        # Test preprocessing
        processed_data = preprocessor.preprocess(self.data, fit=True)
        
        # Check that stress hours are detected
        self.assertIn('stress_hour', processed_data.columns)
        self.assertTrue(processed_data['stress_hour'].dtype in [np.int64, np.int32, bool])
        
        # Check that time features are created
        time_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_winter']
        for feature in time_features:
            self.assertIn(feature, processed_data.columns)
    
    def test_stress_detection(self):
        """Test stress hour detection logic."""
        preprocessor = ImprovedDataPreprocessor(random_state=42)
        processed_data = preprocessor.preprocess(self.data, fit=True)
        
        # Check stress rate is reasonable (not too high or too low)
        stress_rate = processed_data['stress_hour'].mean()
        self.assertGreater(stress_rate, 0.001)  # At least some stress hours
        self.assertLess(stress_rate, 0.5)  # Not more than 50% stress hours
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        # Create data with missing values
        data_with_missing = self.data.copy()
        data_with_missing.loc[data_with_missing.index[:100], 'heat_demand_total'] = np.nan
        
        preprocessor = ImprovedDataPreprocessor(
            imputation_method='knn',
            random_state=42
        )
        
        processed_data = preprocessor.preprocess(data_with_missing, fit=True)
        
        # Check that missing values are handled
        self.assertFalse(processed_data['heat_demand_total'].isnull().any())


if __name__ == '__main__':
    unittest.main()
