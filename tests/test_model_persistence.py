#!/usr/bin/env python3
"""
Test Model Persistence
=====================

Tests for model saving and loading functionality.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.stress_detector import StressClassifier
from models.cost_sensitive_learner import create_cost_sensitive_model


class TestModelPersistence(unittest.TestCase):
    """Test model persistence functionality."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        np.random.seed(42)
        n_samples = 1000
        
        self.X = np.random.randn(n_samples, 10)
        self.y = np.random.binomial(1, 0.1, n_samples)
        self.feature_names = [f'feature_{i}' for i in range(10)]
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_stress_classifier_save_load(self):
        """Test StressClassifier save and load functionality."""
        # Create and train classifier
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
        model_path = Path(self.temp_dir) / 'test_model.joblib'
        classifier.save(str(model_path))
        
        # Check file exists
        self.assertTrue(model_path.exists())
        
        # Load model
        loaded_classifier = StressClassifier.load(str(model_path))
        
        # Test that loaded model works
        predictions = loaded_classifier.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        
        # Test that predictions are consistent
        original_predictions = classifier.predict(self.X)
        np.testing.assert_array_equal(predictions, original_predictions)
    
    def test_model_info_persistence(self):
        """Test that model information is preserved during save/load."""
        # Create classifier
        classifier = StressClassifier(model_type='xgboost', random_state=42)
        classifier.model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        classifier.feature_names = self.feature_names
        classifier.model.fit(self.X, self.y)
        classifier.is_fitted = True
        
        # Get original info
        original_info = classifier.get_model_info()
        
        # Save and load
        model_path = Path(self.temp_dir) / 'test_model.joblib'
        classifier.save(str(model_path))
        loaded_classifier = StressClassifier.load(str(model_path))
        
        # Get loaded info
        loaded_info = loaded_classifier.get_model_info()
        
        # Compare info
        self.assertEqual(original_info['model_type'], loaded_info['model_type'])
        self.assertEqual(original_info['feature_count'], loaded_info['feature_count'])
        self.assertEqual(original_info['is_fitted'], loaded_info['is_fitted'])
    
    def test_feature_names_persistence(self):
        """Test that feature names are preserved during save/load."""
        # Create classifier
        classifier = StressClassifier(model_type='xgboost', random_state=42)
        classifier.model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        classifier.feature_names = self.feature_names
        classifier.model.fit(self.X, self.y)
        classifier.is_fitted = True
        
        # Save and load
        model_path = Path(self.temp_dir) / 'test_model.joblib'
        classifier.save(str(model_path))
        loaded_classifier = StressClassifier.load(str(model_path))
        
        # Check feature names
        self.assertEqual(classifier.feature_names, loaded_classifier.feature_names)
        self.assertEqual(self.feature_names, loaded_classifier.feature_names)
    
    def test_cost_sensitive_model_persistence(self):
        """Test cost-sensitive model persistence."""
        # Create cost-sensitive model
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        # Train model
        model.fit(self.X, self.y)
        
        # Test predictions
        original_predictions = model.predict(self.X)
        original_probabilities = model.predict_proba(self.X)
        
        # Create classifier wrapper
        classifier = StressClassifier(model_type='xgboost', random_state=42)
        classifier.model = model
        classifier.feature_names = self.feature_names
        classifier.is_fitted = True
        
        # Save and load
        model_path = Path(self.temp_dir) / 'test_cost_sensitive_model.joblib'
        classifier.save(str(model_path))
        loaded_classifier = StressClassifier.load(str(model_path))
        
        # Test loaded model
        loaded_predictions = loaded_classifier.predict(self.X)
        loaded_probabilities = loaded_classifier.predict_proba(self.X)
        
        # Compare predictions
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        np.testing.assert_array_almost_equal(original_probabilities, loaded_probabilities, decimal=5)
    
    def test_invalid_save_path(self):
        """Test saving to invalid path."""
        classifier = StressClassifier(model_type='xgboost', random_state=42)
        classifier.model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        classifier.feature_names = self.feature_names
        classifier.model.fit(self.X, self.y)
        classifier.is_fitted = True
        
        # Try to save to invalid path
        invalid_path = Path(self.temp_dir) / 'nonexistent' / 'model.joblib'
        
        # Should create directory and save successfully
        classifier.save(str(invalid_path))
        self.assertTrue(invalid_path.exists())
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file."""
        nonexistent_path = Path(self.temp_dir) / 'nonexistent_model.joblib'
        
        with self.assertRaises(FileNotFoundError):
            StressClassifier.load(str(nonexistent_path))
    
    def test_load_corrupted_file(self):
        """Test loading corrupted file."""
        # Create a corrupted file
        corrupted_path = Path(self.temp_dir) / 'corrupted_model.joblib'
        with open(corrupted_path, 'w') as f:
            f.write("This is not a valid joblib file")
        
        with self.assertRaises(Exception):
            StressClassifier.load(str(corrupted_path))


if __name__ == '__main__':
    unittest.main()
