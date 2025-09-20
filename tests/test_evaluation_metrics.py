#!/usr/bin/env python3
"""
Test Evaluation Metrics
=======================

Tests for evaluation metrics functionality.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from evaluation.metrics import (
    calculate_comprehensive_metrics,
    calculate_gmean,
    calculate_balanced_accuracy,
    calculate_specificity,
    calculate_sensitivity,
    evaluate_model_performance,
    compare_models
)


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create imbalanced test data
        self.y_true = np.random.binomial(1, 0.1, n_samples)
        self.y_pred = np.random.binomial(1, 0.1, n_samples)
        self.y_proba = np.random.uniform(0, 1, n_samples)
    
    def test_basic_metrics(self):
        """Test basic metric calculations."""
        # Test G-mean calculation
        gmean = calculate_gmean(self.y_true, self.y_pred)
        self.assertIsInstance(gmean, float)
        self.assertGreaterEqual(gmean, 0)
        self.assertLessEqual(gmean, 1)
        
        # Test balanced accuracy
        balanced_acc = calculate_balanced_accuracy(self.y_true, self.y_pred)
        self.assertIsInstance(balanced_acc, float)
        self.assertGreaterEqual(balanced_acc, 0)
        self.assertLessEqual(balanced_acc, 1)
        
        # Test specificity
        specificity = calculate_specificity(self.y_true, self.y_pred)
        self.assertIsInstance(specificity, float)
        self.assertGreaterEqual(specificity, 0)
        self.assertLessEqual(specificity, 1)
        
        # Test sensitivity
        sensitivity = calculate_sensitivity(self.y_true, self.y_pred)
        self.assertIsInstance(sensitivity, float)
        self.assertGreaterEqual(sensitivity, 0)
        self.assertLessEqual(sensitivity, 1)
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        metrics = calculate_comprehensive_metrics(
            self.y_true, self.y_pred, self.y_proba
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'gmean',
            'balanced_accuracy', 'specificity', 'sensitivity',
            'auc_roc', 'auc_pr', 'brier_score'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # Check confusion matrix
        self.assertIn('confusion_matrix', metrics)
        cm = metrics['confusion_matrix']
        self.assertIn('tn', cm)
        self.assertIn('fp', cm)
        self.assertIn('fn', cm)
        self.assertIn('tp', cm)
    
    def test_model_performance_evaluation(self):
        """Test model performance evaluation."""
        evaluation = evaluate_model_performance(
            self.y_true, self.y_pred, self.y_proba, "Test Model"
        )
        
        # Check evaluation structure
        self.assertIn('model_name', evaluation)
        self.assertIn('metrics', evaluation)
        self.assertIn('class_distribution', evaluation)
        self.assertIn('data_info', evaluation)
        
        self.assertEqual(evaluation['model_name'], "Test Model")
        
        # Check data info
        data_info = evaluation['data_info']
        self.assertIn('total_samples', data_info)
        self.assertIn('positive_samples', data_info)
        self.assertIn('negative_samples', data_info)
        self.assertIn('positive_rate', data_info)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Create multiple model evaluations
        evaluations = {
            'Model1': evaluate_model_performance(
                self.y_true, self.y_pred, self.y_proba, "Model1"
            ),
            'Model2': evaluate_model_performance(
                self.y_true, self.y_pred, self.y_proba, "Model2"
            )
        }
        
        # Compare models
        comparison_df = compare_models(evaluations, primary_metric='gmean')
        
        # Check comparison structure
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertIn('Model', comparison_df.columns)
        self.assertIn('G-mean', comparison_df.columns)
        self.assertEqual(len(comparison_df), 2)
    
    def test_edge_cases(self):
        """Test edge cases in metrics calculation."""
        # Test with all zeros
        y_zeros = np.zeros(100)
        y_ones = np.ones(100)
        
        # Should not raise errors
        gmean = calculate_gmean(y_zeros, y_ones)
        balanced_acc = calculate_balanced_accuracy(y_zeros, y_ones)
        
        self.assertIsInstance(gmean, float)
        self.assertIsInstance(balanced_acc, float)
        
        # Test with all ones
        gmean = calculate_gmean(y_ones, y_zeros)
        balanced_acc = calculate_balanced_accuracy(y_ones, y_zeros)
        
        self.assertIsInstance(gmean, float)
        self.assertIsInstance(balanced_acc, float)


if __name__ == '__main__':
    unittest.main()
