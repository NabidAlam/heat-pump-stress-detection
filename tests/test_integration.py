#!/usr/bin/env python3
"""
Integration Tests
=================

End-to-end integration tests for the complete pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.improved_preprocessor import ImprovedDataPreprocessor
from models.cost_sensitive_learner import create_cost_sensitive_model
from models.threshold_optimizer import ThresholdOptimizer
from evaluation.metrics import calculate_comprehensive_metrics
from dr.heuristic import HeuristicDRController


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test data and temporary directory."""
        np.random.seed(42)
        n_samples = 2000  # Larger dataset for integration test
        
        # Create realistic synthetic data
        self.data = pd.DataFrame({
            'heat_demand_total': np.random.exponential(100, n_samples),
            'cop_average': np.random.uniform(2, 5, n_samples),
            'cop_ashp': np.random.uniform(2, 5, n_samples)
        })
        self.data.index = pd.date_range('2020-01-01', periods=n_samples, freq='h')
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline(self):
        """Test the complete analysis pipeline."""
        # Step 1: Data preprocessing
        preprocessor = ImprovedDataPreprocessor(
            imputation_method='knn',
            scaling_method='robust',
            random_state=42
        )
        
        processed_data = preprocessor.preprocess(self.data, fit=True)
        
        # Check preprocessing results
        self.assertIn('stress_hour', processed_data.columns)
        self.assertGreater(len(processed_data.columns), len(self.data.columns))
        
        # Step 2: Prepare features and target
        feature_names = preprocessor.get_feature_names()
        X = processed_data[feature_names].values
        y = processed_data['stress_hour'].values
        
        # Split data temporally
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Step 3: Train cost-sensitive model
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Step 4: Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Step 5: Optimize threshold
        threshold_optimizer = ThresholdOptimizer(
            optimization_metrics=['f1', 'gmean'],
            random_state=42
        )
        
        optimal_thresholds = threshold_optimizer.optimize_thresholds(
            model, X_train, y_train
        )
        
        # Step 6: Evaluate with optimized threshold
        recommended_threshold = optimal_thresholds['gmean']
        y_pred_optimized = (y_proba >= recommended_threshold).astype(int)
        
        # Step 7: Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            y_test, y_pred_optimized, y_proba
        )
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['accuracy'], 0.5)
        self.assertGreater(metrics['f1'], 0.0)
        self.assertGreater(metrics['gmean'], 0.0)
        
        # Step 8: DR analysis
        stress_hours = processed_data['stress_hour'] == 1
        dr_controller = HeuristicDRController(
            threshold=0.6,
            shift_fraction=0.1,
            random_state=42
        )
        
        dr_results = dr_controller.apply_dr_strategy(
            processed_data, stress_hours, strategy='conservative'
        )
        
        # Check DR results
        self.assertIn('impact_metrics', dr_results)
        self.assertIn('stress_reduction', dr_results['impact_metrics'])
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline with missing data."""
        # Introduce missing values
        data_with_missing = self.data.copy()
        missing_indices = np.random.choice(
            data_with_missing.index, 
            size=int(len(data_with_missing) * 0.05), 
            replace=False
        )
        data_with_missing.loc[missing_indices, 'heat_demand_total'] = np.nan
        
        # Run preprocessing
        preprocessor = ImprovedDataPreprocessor(
            imputation_method='knn',
            random_state=42
        )
        
        processed_data = preprocessor.preprocess(data_with_missing, fit=True)
        
        # Check that missing values are handled
        self.assertFalse(processed_data['heat_demand_total'].isnull().any())
        
        # Continue with model training
        feature_names = preprocessor.get_feature_names()
        X = processed_data[feature_names].values
        y = processed_data['stress_hour'].values
        
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        self.assertEqual(len(predictions), len(y))
    
    def test_pipeline_with_different_countries(self):
        """Test pipeline with different country data characteristics."""
        # Simulate different countries with different characteristics
        countries = {
            'DE': {'demand_scale': 1.2, 'cop_scale': 0.9},  # Higher demand, lower COP
            'FR': {'demand_scale': 0.8, 'cop_scale': 1.1},  # Lower demand, higher COP
            'SE': {'demand_scale': 1.5, 'cop_scale': 0.8}   # Much higher demand, lower COP
        }
        
        results = {}
        
        for country, params in countries.items():
            # Create country-specific data
            country_data = self.data.copy()
            country_data['heat_demand_total'] *= params['demand_scale']
            country_data['cop_average'] *= params['cop_scale']
            country_data['cop_ashp'] *= params['cop_scale']
            
            # Run preprocessing
            preprocessor = ImprovedDataPreprocessor(random_state=42)
            processed_data = preprocessor.preprocess(country_data, fit=True)
            
            # Calculate stress rate
            stress_rate = processed_data['stress_hour'].mean()
            results[country] = stress_rate
            
            # Check that stress rate is reasonable
            self.assertGreater(stress_rate, 0.001)
            self.assertLess(stress_rate, 0.5)
        
        # Check that different countries have different stress rates
        self.assertNotEqual(results['DE'], results['FR'])
        self.assertNotEqual(results['SE'], results['FR'])
    
    def test_model_persistence_integration(self):
        """Test model persistence in integration context."""
        # Run preprocessing
        preprocessor = ImprovedDataPreprocessor(random_state=42)
        processed_data = preprocessor.preprocess(self.data, fit=True)
        
        # Train model
        feature_names = preprocessor.get_feature_names()
        X = processed_data[feature_names].values
        y = processed_data['stress_hour'].values
        
        model = create_cost_sensitive_model(
            model_type='xgboost',
            cost_ratio=5.0,
            random_state=42
        )
        
        model.fit(X, y)
        
        # Save model
        model_path = Path(self.temp_dir) / 'test_model.joblib'
        from models.stress_detector import StressClassifier
        
        classifier = StressClassifier('xgboost', 42)
        classifier.model = model
        classifier.feature_names = feature_names
        classifier.is_fitted = True
        
        classifier.save(str(model_path))
        
        # Load model and test
        loaded_classifier = StressClassifier.load(str(model_path))
        predictions = loaded_classifier.predict(X)
        
        self.assertEqual(len(predictions), len(y))
        
        # Clean up
        model_path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
