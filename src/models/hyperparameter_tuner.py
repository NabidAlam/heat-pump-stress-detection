#!/usr/bin/env python3
"""
Hyperparameter Tuning for Cost-Sensitive Models
===============================================

This module provides comprehensive hyperparameter tuning capabilities
for cost-sensitive learning models using Optuna optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from loguru import logger
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuner for cost-sensitive models.
    
    Uses Optuna for efficient hyperparameter optimization with
    focus on imbalanced data metrics.
    """
    
    def __init__(self, 
                 model_type: str = 'xgboost',
                 optimization_metric: str = 'gmean',
                 n_trials: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_type: Type of model to tune ('xgboost', 'random_forest')
            optimization_metric: Metric to optimize ('gmean', 'f1', 'balanced_accuracy', 'auc_pr')
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter tuning. Install with: pip install optuna")
        
        self.model_type = model_type.lower()
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.best_params = {}
        self.best_score = 0.0
        self.study = None
        
        logger.info(f"Initialized HyperparameterTuner for {model_type} with {optimization_metric} optimization")
    
    def comprehensive_tuning(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           model_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_type: Override model type
        
        Returns:
            Dictionary of best hyperparameters
        """
        if model_type is not None:
            self.model_type = model_type.lower()
        
        logger.info(f"Starting comprehensive tuning for {self.model_type}")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(trial, X_train, y_train, X_val, y_val)
        
        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials)
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Tuning completed. Best {self.optimization_metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def _objective_function(self, 
                          trial: optuna.Trial,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray) -> float:
        """Objective function for Optuna optimization."""
        # Suggest hyperparameters based on model type
        if self.model_type == 'xgboost':
            params = self._suggest_xgboost_params(trial)
        elif self.model_type == 'random_forest':
            params = self._suggest_random_forest_params(trial)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Create and train model
        model = self._create_model_with_params(params)
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate optimization metric
        score = self._calculate_metric(y_val, y_pred, y_proba, self.optimization_metric)
        
        return score
    
    def _suggest_xgboost_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': self.random_state,
            'eval_metric': 'logloss',
            'verbosity': 0,
            'n_jobs': -1
        }
        
        # Tune cost ratio if enabled
        if trial.suggest_categorical('tune_cost_ratio', [True, False]):
            cost_ratio = trial.suggest_float('cost_ratio', 1.0, 20.0)
            params['scale_pos_weight'] = cost_ratio
        
        return params
    
    def _suggest_random_forest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest Random Forest hyperparameters."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Tune cost ratio if enabled
        if trial.suggest_categorical('tune_cost_ratio', [True, False]):
            cost_ratio = trial.suggest_float('cost_ratio', 1.0, 20.0)
            params['class_weight'] = {0: 1.0, 1: cost_ratio}
        
        return params
    
    def _create_model_with_params(self, params: Dict[str, Any]):
        """Create model with given parameters."""
        if self.model_type == 'xgboost':
            from xgboost import XGBClassifier
            # Remove scale_pos_weight from base_params to avoid conflict
            params_copy = params.copy()
            if 'scale_pos_weight' in params_copy:
                del params_copy['scale_pos_weight']
            
            model = XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                verbosity=0,
                n_jobs=-1,
                scale_pos_weight=params.get('scale_pos_weight', 1.0),
                **params_copy
            )
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            # Remove class_weight from base_params to avoid conflict
            params_copy = params.copy()
            if 'class_weight' in params_copy:
                del params_copy['class_weight']
            
            model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                class_weight=params.get('class_weight', 'balanced'),
                **params_copy
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def _calculate_metric(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_proba: Optional[np.ndarray],
                         metric: str) -> float:
        """Calculate specified metric."""
        if metric == 'gmean':
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return np.sqrt(sensitivity * specificity)
        
        elif metric == 'f1':
            from sklearn.metrics import f1_score
            return f1_score(y_true, y_pred, zero_division=0)
        
        elif metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            return balanced_accuracy_score(y_true, y_pred)
        
        elif metric == 'auc_pr':
            if y_proba is not None:
                from sklearn.metrics import average_precision_score
                return average_precision_score(y_true, y_proba)
            else:
                return 0.0
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def tune_cost_ratio(self, 
                       X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       cost_ratio_range: Tuple[float, float] = (1.0, 20.0)) -> float:
        """
        Tune the cost ratio for cost-sensitive learning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cost_ratio_range: Range of cost ratios to test
        
        Returns:
            Optimal cost ratio
        """
        logger.info("Tuning cost ratio for cost-sensitive learning")
        
        best_cost_ratio = 1.0
        best_score = 0.0
        
        cost_ratios = np.linspace(cost_ratio_range[0], cost_ratio_range[1], 20)
        
        for cost_ratio in cost_ratios:
            # Create model with cost ratio
            if self.model_type == 'xgboost':
                from xgboost import XGBClassifier
                model = XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    verbosity=0,
                    n_jobs=-1,
                    scale_pos_weight=cost_ratio
                )
            elif self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1,
                    class_weight={0: 1, 1: cost_ratio}
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            score = self._calculate_metric(y_val, y_pred, y_proba, self.optimization_metric)
            
            if score > best_score:
                best_score = score
                best_cost_ratio = cost_ratio
        
        logger.info(f"Optimal cost ratio: {best_cost_ratio:.2f} (score: {best_score:.4f})")
        return best_cost_ratio
    
    def save_results(self, filepath: str):
        """Save tuning results to file."""
        import joblib
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_metric': self.optimization_metric,
            'model_type': self.model_type,
            'n_trials': self.n_trials,
            'study': self.study
        }
        
        joblib.dump(results, filepath)
        logger.info(f"Tuning results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load tuning results from file."""
        import joblib
        
        results = joblib.load(filepath)
        self.best_params = results['best_params']
        self.best_score = results['best_score']
        self.optimization_metric = results['optimization_metric']
        self.model_type = results['model_type']
        self.n_trials = results['n_trials']
        self.study = results['study']
        
        logger.info(f"Tuning results loaded from {filepath}")


def create_tuned_model(model_type: str = 'xgboost',
                      optimization_metric: str = 'gmean',
                      n_trials: int = 50,
                      cv_folds: int = 5,
                      random_state: int = 42) -> HyperparameterTuner:
    """
    Create a hyperparameter tuner instance.
    
    Args:
        model_type: Type of model to tune
        optimization_metric: Metric to optimize
        n_trials: Number of optimization trials
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
    
    Returns:
        Configured HyperparameterTuner instance
    """
    return HyperparameterTuner(
        model_type=model_type,
        optimization_metric=optimization_metric,
        n_trials=n_trials,
        cv_folds=cv_folds,
        random_state=random_state
    )
