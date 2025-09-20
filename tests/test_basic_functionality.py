#!/usr/bin/env python3
"""
Basic Functionality Tests
=========================

Basic tests to ensure core functionality works correctly.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.improved_preprocessor import ImprovedDataPreprocessor
from models.cost_sensitive_learner import create_cost_sensitive_model
from models.threshold_optimizer import ThresholdOptimizer
from evaluation.metrics import calculate_comprehensive_metrics


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of core modules."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic test data
        np.random.seed(42)
        n_samples = 1000
        
        self.X = np.random.randn(n_samples, 10)
        self.y = np.random.binomial(1, 0.1, n_samples)  # Imbalanced data
        
        # Create DataFrame for preprocessor tests
        self.data = pd.DataFrame({
            'heat_demand_total': np.random.exponential(100, n_samples),
            'cop_average': np.random.uniform(2, 5, n_samples),
            'cop_ashp': np.random.uniform(2, 5, n_samples)
        })
        self.data.index = pd.date_range('2020-01-01', periods=n_samples, freq='h')
    
    def test_improved_preprocessor(self):
        """Test improved preprocessor functionality."""
        preprocessor = ImprovedDataPreprocessor(
            imputation_method='knn',
            scaling_method='robust',
            random_state=42
        )
        
        # Test preprocessing
        processed_data = preprocessor.preprocess(self.data, fit=True)
        
        # Check that data is processed
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertGreater(len(processed_data.columns), len(self.data.columns))
        self.assertIn('stress_hour', processed_data.columns)
    
    def test_cost_sensitive_learner(self):
        """Test cost-sensitive learner functionality."""
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        # Test training
        model.fit(self.X, self.y)
        
        # Test prediction
        predictions = model.predict(self.X)
        probabilities = model.predict_proba(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertEqual(probabilities.shape, (len(self.y), 2))
    
    def test_threshold_optimizer(self):
        """Test threshold optimizer functionality."""
        optimizer = ThresholdOptimizer(
            optimization_metrics=['f1', 'gmean'],
            random_state=42
        )
        
        # Create a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        model.fit(self.X, self.y)
        
        # Test threshold optimization
        optimal_thresholds = optimizer.optimize_thresholds(model, self.X, self.y)
        
        self.assertIsInstance(optimal_thresholds, dict)
        self.assertIn('f1', optimal_thresholds)
        self.assertIn('gmean', optimal_thresholds)
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        # Create test predictions
        y_pred = np.random.binomial(1, 0.1, len(self.y))
        y_proba = np.random.uniform(0, 1, len(self.y))
        
        # Test metrics calculation
        metrics = calculate_comprehensive_metrics(self.y, y_pred, y_proba)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('gmean', metrics)
        self.assertIn('balanced_accuracy', metrics)


if __name__ == '__main__':
    unittest.main()
