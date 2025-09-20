#!/usr/bin/env python3
"""
Test Stress Detection Models
============================

Tests for stress detection model functionality.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.stress_detector import StressClassifier
from models.cost_sensitive_learner import create_cost_sensitive_model


class TestStressDetector(unittest.TestCase):
    """Test stress detection model functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 1000
        
        self.X = np.random.randn(n_samples, 10)
        self.y = np.random.binomial(1, 0.1, n_samples)  # Imbalanced data
        
        # Create feature names
        self.feature_names = [f'feature_{i}' for i in range(10)]
    
    def test_stress_classifier_creation(self):
        """Test stress classifier creation."""
        classifier = StressClassifier(model_type='xgboost', random_state=42)
        
        self.assertEqual(classifier.model_type, 'xgboost')
        self.assertEqual(classifier.random_state, 42)
        self.assertFalse(classifier.is_fitted)
    
    def test_cost_sensitive_model_creation(self):
        """Test cost-sensitive model creation."""
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.cost_ratio, 5.0)
    
    def test_model_training(self):
        """Test model training functionality."""
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        # Train model
        model.fit(self.X, self.y)
        
        # Test predictions
        predictions = model.predict(self.X)
        probabilities = model.predict_proba(self.X)
        
        self.assertEqual(len(predictions), len(self.y))
        self.assertEqual(probabilities.shape, (len(self.y), 2))
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        # Create and train model
        classifier = StressClassifier(model_type='xgboost', random_state=42)
        classifier.model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        classifier.feature_names = self.feature_names
        classifier.model.fit(self.X, self.y)
        classifier.is_fitted = True
        
        # Save model
        test_path = 'test_model.joblib'
        classifier.save(test_path)
        
        # Load model
        loaded_classifier = StressClassifier.load(test_path)
        
        # Test that loaded model works
        predictions = loaded_classifier.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        
        # Clean up
        Path(test_path).unlink(missing_ok=True)
    
    def test_feature_importances(self):
        """Test feature importance extraction."""
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        model.fit(self.X, self.y)
        
        # Test feature importances
        importances = model.get_feature_importances()
        if importances is not None:
            self.assertEqual(len(importances), self.X.shape[1])
            self.assertTrue(np.all(importances >= 0))


if __name__ == '__main__':
    unittest.main()
