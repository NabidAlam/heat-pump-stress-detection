#!/usr/bin/env python3
"""
Cross-Validation and Model Validation
=====================================

This module provides cross-validation capabilities specifically
designed for time series data and imbalanced datasets.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, average_precision_score)
from loguru import logger
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class CrossValidator:
    """
    Cross-validation for time series and imbalanced data.
    
    Provides specialized cross-validation strategies for energy system data
    with temporal dependencies and class imbalance.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Size of test set
            random_state: Random state for reproducibility
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        
        logger.info(f"Initialized CrossValidator with {n_splits} splits")
    
    def time_series_cv(self, 
                      model: Any, 
                      X: np.ndarray, 
                      y: np.ndarray,
                      metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1']) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            metrics: List of metrics to calculate
        
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("Starting time series cross-validation")
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        cv_results = {metric: [] for metric in metrics}
        cv_results['folds'] = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            y_pred_fold = model.predict(X_val_fold)
            
            # Calculate metrics
            fold_metrics = {}
            for metric in metrics:
                if metric == 'accuracy':
                    score = accuracy_score(y_val_fold, y_pred_fold)
                elif metric == 'precision':
                    score = precision_score(y_val_fold, y_pred_fold, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_val_fold, y_pred_fold, zero_division=0)
                elif metric == 'f1':
                    score = f1_score(y_val_fold, y_pred_fold, zero_division=0)
                elif metric == 'roc_auc':
                    if hasattr(model, 'predict_proba'):
                        y_proba_fold = model.predict_proba(X_val_fold)[:, 1]
                        score = roc_auc_score(y_val_fold, y_proba_fold)
                    else:
                        score = 0.0
                elif metric == 'pr_auc':
                    if hasattr(model, 'predict_proba'):
                        y_proba_fold = model.predict_proba(X_val_fold)[:, 1]
                        score = average_precision_score(y_val_fold, y_proba_fold)
                    else:
                        score = 0.0
                else:
                    logger.warning(f"Unknown metric: {metric}")
                    score = 0.0
                
                cv_results[metric].append(score)
                fold_metrics[metric] = score
            
            cv_results['folds'].append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'metrics': fold_metrics
            })
        
        # Calculate summary statistics
        cv_results['summary'] = {}
        for metric in metrics:
            scores = cv_results[metric]
            cv_results['summary'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        logger.info("Time series cross-validation completed")
        return cv_results
    
    def stratified_cv(self, 
                     model: Any, 
                     X: np.ndarray, 
                     y: np.ndarray,
                     metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1']) -> Dict[str, Any]:
        """
        Perform stratified cross-validation.
        
        Args:
            model: Model to validate
            X: Feature matrix
            y: Target vector
            metrics: List of metrics to calculate
        
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("Starting stratified cross-validation")
        
        # Use StratifiedKFold for imbalanced data
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        cv_results = {metric: [] for metric in metrics}
        cv_results['folds'] = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            y_pred_fold = model.predict(X_val_fold)
            
            # Calculate metrics
            fold_metrics = {}
            for metric in metrics:
                if metric == 'accuracy':
                    score = accuracy_score(y_val_fold, y_pred_fold)
                elif metric == 'precision':
                    score = precision_score(y_val_fold, y_pred_fold, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_val_fold, y_pred_fold, zero_division=0)
                elif metric == 'f1':
                    score = f1_score(y_val_fold, y_pred_fold, zero_division=0)
                elif metric == 'roc_auc':
                    if hasattr(model, 'predict_proba'):
                        y_proba_fold = model.predict_proba(X_val_fold)[:, 1]
                        score = roc_auc_score(y_val_fold, y_proba_fold)
                    else:
                        score = 0.0
                elif metric == 'pr_auc':
                    if hasattr(model, 'predict_proba'):
                        y_proba_fold = model.predict_proba(X_val_fold)[:, 1]
                        score = average_precision_score(y_val_fold, y_proba_fold)
                    else:
                        score = 0.0
                else:
                    logger.warning(f"Unknown metric: {metric}")
                    score = 0.0
                
                cv_results[metric].append(score)
                fold_metrics[metric] = score
            
            cv_results['folds'].append({
                'fold': fold + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'metrics': fold_metrics
            })
        
        # Calculate summary statistics
        cv_results['summary'] = {}
        for metric in metrics:
            scores = cv_results[metric]
            cv_results['summary'][metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        
        logger.info("Stratified cross-validation completed")
        return cv_results
    
    def get_cv_summary(self, cv_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Get cross-validation summary as DataFrame.
        
        Args:
            cv_results: Cross-validation results
        
        Returns:
            DataFrame with CV summary
        """
        summary_data = []
        
        for metric, stats in cv_results['summary'].items():
            summary_data.append({
                'Metric': metric,
                'Mean': stats['mean'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max']
            })
        
        return pd.DataFrame(summary_data)
