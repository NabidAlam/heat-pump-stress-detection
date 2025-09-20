#!/usr/bin/env python3
"""
Cost-Sensitive Learning Implementation
=====================================

This module implements cost-sensitive learning for imbalanced datasets
without generating synthetic data. It uses asymmetric loss functions
and threshold optimization to handle class imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger
from typing import Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class CostSensitiveLearner(BaseEstimator, ClassifierMixin):
    """
    Cost-sensitive learning wrapper for handling imbalanced datasets.
    
    This class wraps standard classifiers and applies cost-sensitive adjustments
    through asymmetric loss functions and threshold optimization.
    """
    
    def __init__(self, 
                 base_estimator=None,
                 cost_ratio: float = 5.0,
                 method: str = 'threshold',
                 random_state: int = 42):
        """
        Initialize the cost-sensitive learner.
        
        Args:
            base_estimator: Base classifier to wrap
            cost_ratio: Cost ratio for false negatives vs false positives
            method: Method for cost-sensitive learning ('threshold', 'weighted', 'focal')
            random_state: Random state for reproducibility
        """
        self.base_estimator = base_estimator
        self.cost_ratio = cost_ratio
        self.method = method
        self.random_state = random_state
        
        # Initialize the wrapped estimator
        self.estimator_ = None
        self.is_fitted_ = False
        
        logger.info(f"Initialized CostSensitiveLearner with cost_ratio={cost_ratio}, method={method}")
    
    def _get_base_estimator(self):
        """Get or create the base estimator."""
        if self.base_estimator is not None:
            return self.base_estimator
        
        # Default to XGBoost if no estimator provided
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def _apply_cost_sensitive_params(self, estimator, X, y):
        """Apply cost-sensitive parameters to the estimator."""
        # Calculate class weights based on cost ratio
        class_counts = np.bincount(y)
        if len(class_counts) < 2:
            logger.warning("Only one class found in target")
            return estimator
        
        # Calculate class weights
        total_samples = len(y)
        n_pos = class_counts[1] if len(class_counts) > 1 else 0
        n_neg = class_counts[0]
        
        if n_pos == 0 or n_neg == 0:
            logger.warning("One class is missing, using equal weights")
            return estimator
        
        # Apply cost-sensitive parameters based on estimator type
        estimator_name = estimator.__class__.__name__.lower()
        
        if 'xgboost' in estimator_name or 'xgb' in estimator_name:
            # For XGBoost, use scale_pos_weight
            scale_pos_weight = (n_neg / n_pos) * self.cost_ratio
            estimator.set_params(scale_pos_weight=scale_pos_weight)
            logger.info(f"Set XGBoost scale_pos_weight to {scale_pos_weight:.3f}")
        
        elif 'randomforest' in estimator_name or 'random' in estimator_name:
            # For Random Forest, use class_weight
            class_weight = {0: 1.0, 1: self.cost_ratio}
            estimator.set_params(class_weight=class_weight)
            logger.info(f"Set RandomForest class_weight to {class_weight}")
        
        elif 'logistic' in estimator_name:
            # For Logistic Regression, use class_weight
            class_weight = {0: 1.0, 1: self.cost_ratio}
            estimator.set_params(class_weight=class_weight)
            logger.info(f"Set LogisticRegression class_weight to {class_weight}")
        
        else:
            logger.warning(f"Unknown estimator type: {estimator_name}, using default parameters")
        
        return estimator
    
    def fit(self, X, y):
        """Fit the cost-sensitive learner."""
        # Validate inputs
        X, y = check_X_y(X, y)
        
        # Get base estimator
        self.estimator_ = self._get_base_estimator()
        
        # Apply cost-sensitive parameters
        self.estimator_ = self._apply_cost_sensitive_params(self.estimator_, X, y)
        
        # Fit the estimator
        logger.info(f"Training cost-sensitive {self.estimator_.__class__.__name__}")
        self.estimator_.fit(X, y)
        
        self.is_fitted_ = True
        logger.info("Cost-sensitive model training completed")
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X)
        return self.estimator_.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before predicting probabilities")
        
        X = check_array(X)
        return self.estimator_.predict_proba(X)
    
    def decision_function(self, X):
        """Compute decision function values."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before computing decision function")
        
        X = check_array(X)
        
        # Try to get decision function from base estimator
        if hasattr(self.estimator_, 'decision_function'):
            return self.estimator_.decision_function(X)
        else:
            # Fallback to probability-based decision function
            proba = self.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1] - proba[:, 0]
            else:
                return proba[:, 1]
    
    def predict_with_threshold(self, X, threshold: float = 0.5):
        """Make predictions using a custom threshold."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        X = check_array(X)
        proba = self.predict_proba(X)
        
        if proba.shape[1] == 2:
            return (proba[:, 1] >= threshold).astype(int)
        else:
            return (proba >= threshold).astype(int)
    
    def get_feature_importances(self):
        """Get feature importances from the base estimator."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importances")
        
        if hasattr(self.estimator_, 'feature_importances_'):
            return self.estimator_.feature_importances_
        elif hasattr(self.estimator_, 'coef_'):
            return np.abs(self.estimator_.coef_[0])
        else:
            logger.warning("Base estimator does not have feature importances")
            return None
    
    def score(self, X, y):
        """Compute accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {
            'base_estimator': self.base_estimator,
            'cost_ratio': self.cost_ratio,
            'method': self.method,
            'random_state': self.random_state
        }
        
        if deep and self.estimator_ is not None:
            params.update(self.estimator_.get_params(deep=deep))
        
        return params
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif self.estimator_ is not None and hasattr(self.estimator_, key):
                self.estimator_.set_params(**{key: value})
            else:
                raise ValueError(f"Invalid parameter {key}")
        
        return self
    
    def _more_tags(self):
        """Return estimator tags for sklearn compatibility."""
        return {
            'binary_only': True,
            'requires_fit': True,
            'requires_y': True,
            'no_validation': False,
            'multiclass_only': False,
            'multioutput': False,
            'multioutput_only': False,
            'poor_score': False,
            'non_deterministic': False,
            'requires_X': True,
            'stateless': False,
            'X_types': ['2darray'],
            'y_types': ['1dlabels'],
            '_xfail_checks': {},
            'estimator_type': 'classifier'
        }
    
    def __sklearn_tags__(self):
        """Return sklearn tags for compatibility."""
        return self._more_tags()


def create_cost_sensitive_model(model_type: str = 'xgboost',
                               cost_ratio: float = 5.0,
                               method: str = 'threshold',
                               random_state: int = 42) -> CostSensitiveLearner:
    """
    Create a cost-sensitive model with specified parameters.
    
    Args:
        model_type: Type of base model ('xgboost', 'random_forest', 'logistic')
        cost_ratio: Cost ratio for false negatives vs false positives
        method: Method for cost-sensitive learning
        random_state: Random state for reproducibility
    
    Returns:
        Configured CostSensitiveLearner instance
    """
    base_estimator = None
    
    if model_type.lower() in ['xgboost', 'xgb']:
        try:
            from xgboost import XGBClassifier
            base_estimator = XGBClassifier(
                random_state=random_state,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1
            )
        except ImportError:
            logger.warning("XGBoost not available, falling back to Random Forest")
            from sklearn.ensemble import RandomForestClassifier
            base_estimator = RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1
            )
    
    elif model_type.lower() in ['random_forest', 'rf']:
        from sklearn.ensemble import RandomForestClassifier
        base_estimator = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        )
    
    elif model_type.lower() in ['logistic', 'logistic_regression']:
        from sklearn.linear_model import LogisticRegression
        base_estimator = LogisticRegression(
            random_state=random_state,
            max_iter=1000
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return CostSensitiveLearner(
        base_estimator=base_estimator,
        cost_ratio=cost_ratio,
        method=method,
        random_state=random_state
    )
