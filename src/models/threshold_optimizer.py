#!/usr/bin/env python3
"""
Threshold Optimization for Imbalanced Classification
====================================================

This module provides threshold optimization capabilities for handling
imbalanced datasets by finding optimal decision thresholds based on
various metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score, 
                           roc_auc_score, precision_recall_curve, 
                           roc_curve, confusion_matrix)
from sklearn.model_selection import cross_val_score
from loguru import logger
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ThresholdOptimizer:
    """
    Optimize classification thresholds for imbalanced datasets.
    
    This class finds optimal decision thresholds based on various metrics
    to improve performance on imbalanced data without modifying the data.
    """
    
    def __init__(self, 
                 optimization_metrics: List[str] = ['f1', 'gmean', 'balanced_accuracy'],
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the threshold optimizer.
        
        Args:
            optimization_metrics: List of metrics to optimize for
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.optimization_metrics = optimization_metrics
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.optimization_results = {}
        
        logger.info(f"Initialized ThresholdOptimizer with metrics: {optimization_metrics}")
    
    def calculate_gmean(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate geometric mean of sensitivity and specificity."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(sensitivity * specificity)
    
    def calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return (sensitivity + specificity) / 2
    
    def calculate_youden_j(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Youden's J statistic."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sensitivity + specificity - 1
    
    def optimize_threshold_for_metric(self, 
                                    y_true: np.ndarray, 
                                    y_proba: np.ndarray, 
                                    metric: str) -> Tuple[float, float]:
        """
        Find optimal threshold for a specific metric.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            metric: Metric to optimize ('f1', 'gmean', 'balanced_accuracy', 'youden')
        
        Returns:
            Tuple of (optimal_threshold, best_score)
        """
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # Calculate metric for each threshold
        scores = []
        valid_thresholds = []
        
        for i, threshold in enumerate(thresholds):
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'gmean':
                score = self.calculate_gmean(y_true, y_pred)
            elif metric == 'balanced_accuracy':
                score = self.calculate_balanced_accuracy(y_true, y_pred)
            elif metric == 'youden':
                score = self.calculate_youden_j(y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
            valid_thresholds.append(threshold)
        
        # Find best threshold
        if not scores:
            return 0.5, 0.0
        
        best_idx = np.argmax(scores)
        optimal_threshold = valid_thresholds[best_idx]
        best_score = scores[best_idx]
        
        return optimal_threshold, best_score
    
    def optimize_thresholds(self, 
                          model: Any, 
                          X: np.ndarray, 
                          y: np.ndarray) -> Dict[str, float]:
        """
        Optimize thresholds for multiple metrics using cross-validation.
        
        Args:
            model: Trained classifier with predict_proba method
            X: Feature matrix
            y: Target vector
        
        Returns:
            Dictionary of optimal thresholds for each metric
        """
        from sklearn.model_selection import KFold
        
        logger.info(f"Optimizing thresholds for {len(self.optimization_metrics)} metrics")
        
        # Initialize results
        threshold_results = {metric: [] for metric in self.optimization_metrics}
        score_results = {metric: [] for metric in self.optimization_metrics}
        
        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model on fold
            model.fit(X_train_fold, y_train_fold)
            
            # Get probabilities
            y_proba_fold = model.predict_proba(X_val_fold)[:, 1]
            
            # Optimize for each metric
            for metric in self.optimization_metrics:
                threshold, score = self.optimize_threshold_for_metric(
                    y_val_fold, y_proba_fold, metric
                )
                threshold_results[metric].append(threshold)
                score_results[metric].append(score)
        
        # Calculate average optimal thresholds
        optimal_thresholds = {}
        for metric in self.optimization_metrics:
            optimal_thresholds[metric] = np.mean(threshold_results[metric])
            logger.info(f"{metric}: optimal threshold = {optimal_thresholds[metric]:.3f}, "
                       f"avg score = {np.mean(score_results[metric]):.3f}")
        
        # Store results
        self.optimization_results = {
            'thresholds': threshold_results,
            'scores': score_results,
            'optimal_thresholds': optimal_thresholds
        }
        
        return optimal_thresholds
    
    def evaluate_threshold(self, 
                          y_true: np.ndarray, 
                          y_proba: np.ndarray, 
                          threshold: float) -> Dict[str, float]:
        """
        Evaluate model performance at a specific threshold.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            threshold: Decision threshold
        
        Returns:
            Dictionary of performance metrics
        """
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': (y_pred == y_true).mean(),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'gmean': self.calculate_gmean(y_true, y_pred),
            'balanced_accuracy': self.calculate_balanced_accuracy(y_true, y_pred),
            'youden_j': self.calculate_youden_j(y_true, y_pred)
        }
        
        return metrics
    
    def get_threshold_analysis(self, 
                              y_true: np.ndarray, 
                              y_proba: np.ndarray) -> pd.DataFrame:
        """
        Get comprehensive threshold analysis.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
        
        Returns:
            DataFrame with threshold analysis
        """
        # Generate threshold range
        thresholds = np.linspace(0.1, 0.9, 81)
        
        results = []
        for threshold in thresholds:
            metrics = self.evaluate_threshold(y_true, y_proba, threshold)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def plot_threshold_analysis(self, 
                               y_true: np.ndarray, 
                               y_proba: np.ndarray,
                               save_path: Optional[str] = None):
        """
        Plot threshold analysis.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities for positive class
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Get threshold analysis
        df = self.get_threshold_analysis(y_true, y_proba)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Threshold Optimization Analysis', fontsize=16)
        
        # Plot metrics vs threshold
        metrics_to_plot = ['f1', 'gmean', 'balanced_accuracy', 'youden_j']
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            ax.plot(df['threshold'], df[metric], 'b-', linewidth=2)
            ax.set_xlabel('Threshold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} vs Threshold')
            ax.grid(True, alpha=0.3)
            
            # Mark optimal threshold
            if metric in self.optimization_results.get('optimal_thresholds', {}):
                optimal_threshold = self.optimization_results['optimal_thresholds'][metric]
                optimal_score = df[df['threshold'] == optimal_threshold][metric].iloc[0]
                ax.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7)
                ax.plot(optimal_threshold, optimal_score, 'ro', markersize=8)
                ax.annotate(f'Optimal: {optimal_threshold:.3f}', 
                           xy=(optimal_threshold, optimal_score),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved to {save_path}")
        
        plt.show()
