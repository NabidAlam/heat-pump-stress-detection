#!/usr/bin/env python3
"""
Stress Detection Model Wrapper
==============================

This module provides a wrapper for stress detection models with
standardized interface and persistence capabilities.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class StressClassifier:
    """
    Wrapper class for stress detection models.
    
    Provides a standardized interface for different model types
    and handles model persistence.
    """
    
    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """
        Initialize the stress classifier.
        
        Args:
            model_type: Type of model to use
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        
        logger.info(f"Initialized StressClassifier with model_type={model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Model fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting probabilities")
        
        return self.model.predict_proba(X)
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Get feature importances."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importances")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            logger.warning("Model does not have feature importances")
            return None
    
    def save(self, filepath: str):
        """Save the model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StressClassifier':
        """Load a model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(
            model_type=model_data['model_type'],
            random_state=model_data['random_state']
        )
        
        # Restore attributes
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'random_state': self.random_state
        }
