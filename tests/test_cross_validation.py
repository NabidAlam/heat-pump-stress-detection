#!/usr/bin/env python3
"""
Test Cross-Validation Functionality
===================================

Tests for cross-validation functionality.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from evaluation.validation import CrossValidator
from models.cost_sensitive_learner import create_cost_sensitive_model


class TestCrossValidation(unittest.TestCase):
    """Test cross-validation functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 1000
        
        self.X = np.random.randn(n_samples, 10)
        self.y = np.random.binomial(1, 0.1, n_samples)  # Imbalanced data
    
    def test_cross_validator_creation(self):
        """Test cross-validator creation."""
        cv = CrossValidator(
            n_splits=5,
            test_size=0.2,
            random_state=42
        )
        
        self.assertEqual(cv.n_splits, 5)
        self.assertEqual(cv.test_size, 0.2)
        self.assertEqual(cv.random_state, 42)
    
    def test_time_series_cv(self):
        """Test time series cross-validation."""
        cv = CrossValidator(n_splits=3, random_state=42)
        
        # Create a simple model for testing
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        # Run time series CV
        cv_results = cv.time_series_cv(
            model, self.X, self.y,
            metrics=['accuracy', 'precision', 'recall', 'f1']
        )
        
        # Check results structure
        self.assertIn('accuracy', cv_results)
        self.assertIn('precision', cv_results)
        self.assertIn('recall', cv_results)
        self.assertIn('f1', cv_results)
        self.assertIn('folds', cv_results)
        self.assertIn('summary', cv_results)
        
        # Check that we have results for each fold
        self.assertEqual(len(cv_results['accuracy']), 3)
        self.assertEqual(len(cv_results['folds']), 3)
        
        # Check summary statistics
        summary = cv_results['summary']
        self.assertIn('accuracy', summary)
        self.assertIn('mean', summary['accuracy'])
        self.assertIn('std', summary['accuracy'])
    
    def test_stratified_cv(self):
        """Test stratified cross-validation."""
        cv = CrossValidator(n_splits=3, random_state=42)
        
        # Create a simple model for testing
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        # Run stratified CV
        cv_results = cv.stratified_cv(
            model, self.X, self.y,
            metrics=['accuracy', 'f1', 'roc_auc']
        )
        
        # Check results structure
        self.assertIn('accuracy', cv_results)
        self.assertIn('f1', cv_results)
        self.assertIn('roc_auc', cv_results)
        self.assertIn('folds', cv_results)
        self.assertIn('summary', cv_results)
        
        # Check that we have results for each fold
        self.assertEqual(len(cv_results['accuracy']), 3)
        self.assertEqual(len(cv_results['folds']), 3)
    
    def test_cv_summary(self):
        """Test CV summary generation."""
        cv = CrossValidator(n_splits=3, random_state=42)
        
        # Create mock CV results
        cv_results = {
            'accuracy': [0.8, 0.85, 0.82],
            'f1': [0.6, 0.65, 0.62],
            'summary': {
                'accuracy': {'mean': 0.823, 'std': 0.025, 'min': 0.8, 'max': 0.85},
                'f1': {'mean': 0.623, 'std': 0.025, 'min': 0.6, 'max': 0.65}
            }
        }
        
        # Generate summary DataFrame
        summary_df = cv.get_cv_summary(cv_results)
        
        # Check DataFrame structure
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertIn('Metric', summary_df.columns)
        self.assertIn('Mean', summary_df.columns)
        self.assertIn('Std', summary_df.columns)
        self.assertIn('Min', summary_df.columns)
        self.assertIn('Max', summary_df.columns)
        
        # Check that we have rows for each metric
        self.assertEqual(len(summary_df), 2)
        self.assertIn('accuracy', summary_df['Metric'].values)
        self.assertIn('f1', summary_df['Metric'].values)
    
    def test_different_metrics(self):
        """Test CV with different metric combinations."""
        cv = CrossValidator(n_splits=2, random_state=42)
        
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        # Test with different metric sets
        metrics_sets = [
            ['accuracy'],
            ['accuracy', 'f1'],
            ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
        ]
        
        for metrics in metrics_sets:
            cv_results = cv.time_series_cv(model, self.X, self.y, metrics=metrics)
            
            # Check that all requested metrics are present
            for metric in metrics:
                self.assertIn(metric, cv_results)
                self.assertEqual(len(cv_results[metric]), 2)  # 2 folds
    
    def test_edge_cases(self):
        """Test edge cases in cross-validation."""
        cv = CrossValidator(n_splits=2, random_state=42)
        
        # Test with very small dataset
        X_small = self.X[:10]
        y_small = self.y[:10]
        
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        # Should not raise errors
        cv_results = cv.time_series_cv(
            model, X_small, y_small, metrics=['accuracy']
        )
        
        self.assertIn('accuracy', cv_results)
        self.assertEqual(len(cv_results['accuracy']), 2)


if __name__ == '__main__':
    unittest.main()
