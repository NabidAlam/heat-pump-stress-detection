#!/usr/bin/env python3
"""
Comprehensive Metrics for Imbalanced Classification
==================================================

This module provides comprehensive evaluation metrics specifically
designed for imbalanced datasets and energy system stress detection.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, average_precision_score, confusion_matrix,
                           classification_report, brier_score_loss)
from loguru import logger
from typing import Dict, Any, Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def calculate_gmean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate geometric mean of sensitivity and specificity."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(sensitivity * specificity)
    else:
        # Handle case where only one class is present
        return 0.0


def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate balanced accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return (sensitivity + specificity) / 2
    else:
        # Handle case where only one class is present
        return 0.0


def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate specificity (true negative rate)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # Handle case where only one class is present
        return 0.0


def calculate_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate sensitivity (true positive rate)."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    else:
        # Handle case where only one class is present
        return 0.0


def calculate_youden_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Youden's J statistic."""
    sensitivity = calculate_sensitivity(y_true, y_pred)
    specificity = calculate_specificity(y_true, y_pred)
    return sensitivity + specificity - 1


def calculate_comprehensive_metrics(y_true: np.ndarray, 
                                  y_pred: np.ndarray, 
                                  y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for imbalanced classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class (optional)
    
    Returns:
        Dictionary of comprehensive metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Imbalanced data specific metrics
    metrics['gmean'] = calculate_gmean(y_true, y_pred)
    metrics['balanced_accuracy'] = calculate_balanced_accuracy(y_true, y_pred)
    metrics['specificity'] = calculate_specificity(y_true, y_pred)
    metrics['sensitivity'] = calculate_sensitivity(y_true, y_pred)
    metrics['youden_j'] = calculate_youden_j(y_true, y_pred)
    
    # Probability-based metrics (if probabilities provided)
    if y_proba is not None:
        # Check if both classes are present for AUC calculation
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_proba)
        else:
            # If only one class present, set AUC to 0.5 (random)
            metrics['auc_roc'] = 0.5
            metrics['auc_pr'] = 0.5
        metrics['brier_score'] = brier_score_loss(y_true, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # Additional derived metrics
        tn, fp, fn, tp = cm.ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    else:
        # Handle case where only one class is present
        metrics['confusion_matrix'] = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
        metrics['false_positive_rate'] = 0
        metrics['false_negative_rate'] = 0
        metrics['positive_predictive_value'] = 0
        metrics['negative_predictive_value'] = 0
    
    return metrics


def calculate_class_distribution(y: np.ndarray) -> Dict[str, Any]:
    """Calculate class distribution statistics."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    distribution = {}
    for class_label, count in zip(unique, counts):
        distribution[f'class_{int(class_label)}'] = {
            'count': int(count),
            'percentage': count / total * 100
        }
    
    # Calculate imbalance ratio
    if len(unique) == 2:
        minority_count = min(counts)
        majority_count = max(counts)
        imbalance_ratio = majority_count / minority_count
        distribution['imbalance_ratio'] = imbalance_ratio
        distribution['minority_class_rate'] = minority_count / total
    
    return distribution


def evaluate_model_performance(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             y_proba: Optional[np.ndarray] = None,
                             model_name: str = "Model") -> Dict[str, Any]:
    """
    Comprehensive model performance evaluation.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class (optional)
        model_name: Name of the model for reporting
    
    Returns:
        Dictionary with comprehensive evaluation results
    """
    logger.info(f"Evaluating {model_name} performance")
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_proba)
    
    # Calculate class distribution
    class_dist = calculate_class_distribution(y_true)
    
    # Create evaluation summary
    evaluation = {
        'model_name': model_name,
        'metrics': metrics,
        'class_distribution': class_dist,
        'data_info': {
            'total_samples': len(y_true),
            'positive_samples': int(np.sum(y_true)),
            'negative_samples': int(len(y_true) - np.sum(y_true)),
            'positive_rate': float(np.mean(y_true))
        }
    }
    
    # Log key metrics
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-score: {metrics['f1']:.4f}")
    logger.info(f"  G-mean: {metrics['gmean']:.4f}")
    logger.info(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    
    if y_proba is not None:
        logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    
    return evaluation


def compare_models(evaluations: Dict[str, Dict[str, Any]], 
                  primary_metric: str = 'gmean') -> pd.DataFrame:
    """
    Compare multiple model evaluations.
    
    Args:
        evaluations: Dictionary of model evaluations
        primary_metric: Primary metric for comparison
    
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, evaluation in evaluations.items():
        metrics = evaluation['metrics']
        row = {
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-score': metrics['f1'],
            'G-mean': metrics['gmean'],
            'Balanced Accuracy': metrics['balanced_accuracy'],
            'Specificity': metrics['specificity'],
            'Sensitivity': metrics['sensitivity']
        }
        
        if 'auc_roc' in metrics:
            row['AUC-ROC'] = metrics['auc_roc']
        if 'auc_pr' in metrics:
            row['AUC-PR'] = metrics['auc_pr']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by primary metric
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False)
    
    return df


def generate_classification_report(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 target_names: Optional[List[str]] = None) -> str:
    """Generate detailed classification report."""
    if target_names is None:
        target_names = ['No Stress', 'Stress']
    
    return classification_report(y_true, y_pred, target_names=target_names, output_dict=False)
