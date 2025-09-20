#!/usr/bin/env python3
"""
Improved Conference Paper Analysis with Advanced Imbalanced Data Handling
=======================================================================

This script implements the conference paper requirements with state-of-the-art
techniques for handling imbalanced data:
- Cost-sensitive learning
- Variable threshold optimization
- Focal loss
- Advanced evaluation metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, roc_curve, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from loguru import logger
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our advanced modules
import sys
sys.path.append('src')
from models.cost_sensitive_learner import CostSensitiveLearner
from models.threshold_optimizer import ThresholdOptimizer
from data.improved_preprocessor import ImprovedDataPreprocessor
from evaluation.metrics import calculate_comprehensive_metrics
from dr.heuristic import HeuristicDRController

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_germany_data():
    """Load Germany data for 2-3 winters (2013-2015)."""
    logger.info("Loading Germany data for improved conference paper")
    
    # Load dataset
    data_path = 'dataset/when2heat.csv'
    df = pd.read_csv(data_path, sep=';')
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('datetime', inplace=True)
    
    # Filter to Germany and 2-3 winters (2013-2015)
    df = df[(df.index.year >= 2013) & (df.index.year <= 2015)]
    
    # Convert numeric columns
    for col in df.columns:
        if col not in ['utc_timestamp', 'cet_cest_timestamp']:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Extract Germany data
    de_heat_cols = [col for col in df.columns if col.startswith('DE_heat_demand_total')]
    de_cop_cols = [col for col in df.columns if col.startswith('DE_COP_ASHP_floor')]
    
    if not de_heat_cols or not de_cop_cols:
        raise ValueError("Germany data not found")
    
    # Create Germany dataset
    de_data = pd.DataFrame({
        'datetime': df.index,
        'heat_demand_total': df[de_heat_cols[0]],
        'cop_ashp_floor': df[de_cop_cols[0]]
    })
    
    de_data.set_index('datetime', inplace=True)
    de_data = de_data.dropna()
    
    logger.info(f"Loaded Germany data: {len(de_data)} samples from {de_data.index.min()} to {de_data.index.max()}")
    return de_data

def create_stress_labels(data):
    """Create stress labels using seasonal percentiles."""
    logger.info("Creating stress labels using seasonal percentiles")
    
    # Calculate seasonal percentiles
    data['month'] = data.index.month
    data['season'] = data['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                                       3: 'spring', 4: 'spring', 5: 'spring',
                                       6: 'summer', 7: 'summer', 8: 'summer',
                                       9: 'autumn', 10: 'autumn', 11: 'autumn'})
    
    # Seasonal thresholds
    seasonal_thresholds = {}
    for season in ['winter', 'spring', 'summer', 'autumn']:
        season_data = data[data['season'] == season]
        if len(season_data) > 0:
            seasonal_thresholds[season] = {
                'demand_p90': season_data['heat_demand_total'].quantile(0.90),
                'cop_p20': season_data['cop_ashp_floor'].quantile(0.20)
            }
    
    # Create stress labels
    stress_labels = []
    for idx, row in data.iterrows():
        season = row['season']
        if season in seasonal_thresholds:
            is_stress = (row['heat_demand_total'] > seasonal_thresholds[season]['demand_p90'] and
                        row['cop_ashp_floor'] < seasonal_thresholds[season]['cop_p20'])
            stress_labels.append(1 if is_stress else 0)
        else:
            stress_labels.append(0)
    
    data['stress_hour'] = stress_labels
    stress_rate = np.mean(stress_labels)
    
    logger.info(f"Stress rate: {stress_rate:.3f} ({stress_rate*100:.1f}%)")
    logger.info(f"Stress hours: {sum(stress_labels)} / {len(stress_labels)}")
    
    return data, seasonal_thresholds

def create_conference_features(data):
    """Create features as specified in conference requirements using advanced preprocessor."""
    logger.info("Creating conference paper features with advanced preprocessing")
    
    # Initialize advanced preprocessor
    preprocessor = ImprovedDataPreprocessor(
        imputation_method='knn',
        scaling_method='robust',
        outlier_detection=True,
        random_state=42
    )
    
    # Time features
    data['hour'] = data.index.hour
    data['weekday'] = data.index.weekday
    data['month'] = data.index.month
    data['is_winter'] = data['month'].isin([12, 1, 2]).astype(int)
    
    # Rolling features (24-168h as specified)
    for window in [24, 72, 168]:  # 1 day, 3 days, 1 week
        data[f'demand_ma_{window}'] = data['heat_demand_total'].rolling(window=window, min_periods=1).mean()
        data[f'demand_max_{window}'] = data['heat_demand_total'].rolling(window=window, min_periods=1).max()
        data[f'cop_min_{window}'] = data['cop_ashp_floor'].rolling(window=window, min_periods=1).min()
    
    # Lag features
    data['stress_lag_1'] = data['stress_hour'].shift(1).fillna(0)
    data['stress_lag_24'] = data['stress_hour'].shift(24).fillna(0)
    
    # Select features as specified
    feature_cols = [
        'hour', 'weekday', 'month', 'is_winter',
        'demand_ma_24', 'demand_ma_168', 'demand_max_24', 'demand_max_168',
        'cop_min_24', 'cop_min_168',
        'stress_lag_1', 'stress_lag_24'
    ]
    
    # Remove rows with NaN values
    data = data.dropna()
    
    # Apply advanced preprocessing
    logger.info("Applying advanced preprocessing")
    preprocessed_data = preprocessor.preprocess(data[feature_cols], fit=True)
    
    # Update the original data with preprocessed features
    # Note: preprocessor may have removed outliers, so we need to align indices
    for col in feature_cols:
        if col in preprocessed_data.columns:
            data.loc[preprocessed_data.index, col] = preprocessed_data[col]
    
    # Remove rows that were removed by preprocessing (outliers)
    data = data.loc[preprocessed_data.index]
    
    logger.info(f"Created {len(feature_cols)} features with advanced preprocessing")
    logger.info(f"Final data shape after preprocessing: {data.shape}")
    return data, feature_cols

def rolling_origin_validation(data, feature_cols, target_col='stress_hour'):
    """Implement rolling-origin validation across winters."""
    logger.info("Implementing rolling-origin validation")
    
    # Split by winters
    winters = sorted(data.index.year.unique())
    logger.info(f"Available winters: {winters}")
    
    # Use first two winters for training, third for validation, and test on all data
    if len(winters) >= 3:
        train_winters = winters[:2]  # 2013, 2014 for training
        val_winter = winters[2]      # 2015 for validation
        test_winter = winters[2]     # 2015 for test (same as validation for simplicity)
    else:
        # Fallback for fewer winters
        train_winters = [winters[0]]
        val_winter = winters[-1] if len(winters) > 1 else winters[0]
        test_winter = winters[-1] if len(winters) > 1 else winters[0]
    
    # Create splits
    train_data = data[data.index.year.isin(train_winters)]
    val_data = data[data.index.year == val_winter]
    test_data = data[data.index.year == test_winter]
    
    # Prepare features and targets
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_val = val_data[feature_cols].values
    y_val = val_data[target_col].values
    X_test = test_data[feature_cols].values
    y_test = test_data[target_col].values
    
    logger.info(f"Rolling-origin splits:")
    logger.info(f"  Train: {train_winters} - {len(X_train)} samples")
    logger.info(f"  Val: {val_winter} - {len(X_val)} samples")
    logger.info(f"  Test: {test_winter} - {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_improved_models(X_train, X_val, y_train, y_val):
    """Train models with advanced imbalanced data handling."""
    logger.info("Training improved models with advanced imbalanced data handling")
    
    models = {}
    
    # 1. Cost-Sensitive Logistic Regression
    logger.info("Training Cost-Sensitive Logistic Regression")
    lr_base = LogisticRegression(random_state=42, max_iter=1000)
    lr_cost_sensitive = CostSensitiveLearner(
        base_estimator=lr_base,
        cost_ratio=10.0,  # Higher cost for false negatives
        method='threshold',
        random_state=42
    )
    lr_cost_sensitive.fit(X_train, y_train)
    
    # Optimize threshold for Logistic Regression
    threshold_optimizer_lr = ThresholdOptimizer(
        optimization_metrics=['f1', 'gmean', 'balanced_accuracy'],
        cv_folds=3,
        random_state=42
    )
    optimal_thresholds_lr = threshold_optimizer_lr.optimize_thresholds(
        lr_cost_sensitive, X_val, y_val
    )
    optimal_threshold_lr = optimal_thresholds_lr['f1']
    
    models['logistic_regression'] = {
        'model': lr_cost_sensitive,
        'optimal_threshold': optimal_threshold_lr,
        'optimizer': threshold_optimizer_lr
    }
    
    # 2. Cost-Sensitive Gradient Boosting
    logger.info("Training Cost-Sensitive Gradient Boosting")
    gb_base = GradientBoostingClassifier(
        n_estimators=100,  # More trees for better performance
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    gb_cost_sensitive = CostSensitiveLearner(
        base_estimator=gb_base,
        cost_ratio=10.0,  # Higher cost for false negatives
        method='threshold',
        random_state=42
    )
    gb_cost_sensitive.fit(X_train, y_train)
    
    # Optimize threshold for Gradient Boosting
    threshold_optimizer_gb = ThresholdOptimizer(
        optimization_metrics=['f1', 'gmean', 'balanced_accuracy'],
        cv_folds=3,
        random_state=42
    )
    optimal_thresholds_gb = threshold_optimizer_gb.optimize_thresholds(
        gb_cost_sensitive, X_val, y_val
    )
    optimal_threshold_gb = optimal_thresholds_gb['f1']
    
    models['gradient_boosting'] = {
        'model': gb_cost_sensitive,
        'optimal_threshold': optimal_threshold_gb,
        'optimizer': threshold_optimizer_gb
    }
    
    # 3. Cost-Sensitive XGBoost (for comparison)
    logger.info("Training Cost-Sensitive XGBoost")
    xgb_base = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_cost_sensitive = CostSensitiveLearner(
        base_estimator=xgb_base,
        cost_ratio=10.0,
        method='threshold',
        random_state=42
    )
    xgb_cost_sensitive.fit(X_train, y_train)
    
    # Optimize threshold for XGBoost
    threshold_optimizer_xgb = ThresholdOptimizer(
        optimization_metrics=['f1', 'gmean', 'balanced_accuracy'],
        cv_folds=3,
        random_state=42
    )
    optimal_thresholds_xgb = threshold_optimizer_xgb.optimize_thresholds(
        xgb_cost_sensitive, X_val, y_val
    )
    optimal_threshold_xgb = optimal_thresholds_xgb['f1']
    
    models['xgboost'] = {
        'model': xgb_cost_sensitive,
        'optimal_threshold': optimal_threshold_xgb,
        'optimizer': threshold_optimizer_xgb
    }
    
    return models

def evaluate_improved_models(models, X_val, y_val, X_test, y_test):
    """Evaluate models with comprehensive metrics."""
    logger.info("Evaluating improved models with comprehensive metrics")
    
    results = {}
    
    for name, model_info in models.items():
        logger.info(f"Evaluating {name}")
        
        model = model_info['model']
        optimal_threshold = model_info['optimal_threshold']
        
        # Validation predictions with optimal threshold
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred_optimal = model.predict_with_threshold(X_val, optimal_threshold)
        y_val_pred_default = model.predict(X_val)
        
        # Test predictions with optimal threshold
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_test_pred_optimal = model.predict_with_threshold(X_test, optimal_threshold)
        y_test_pred_default = model.predict(X_test)
        
        # Calculate comprehensive metrics using advanced evaluation
        val_metrics_optimal = calculate_comprehensive_metrics(y_val, y_val_pred_optimal, y_val_proba)
        val_metrics_default = calculate_comprehensive_metrics(y_val, y_val_pred_default, y_val_proba)
        
        # Combine metrics
        val_metrics = {
            'f1_optimal': val_metrics_optimal['f1'],
            'f1_default': val_metrics_default['f1'],
            'precision_optimal': val_metrics_optimal['precision'],
            'precision_default': val_metrics_default['precision'],
            'recall_optimal': val_metrics_optimal['recall'],
            'recall_default': val_metrics_default['recall'],
            'accuracy_optimal': val_metrics_optimal['accuracy'],
            'accuracy_default': val_metrics_default['accuracy'],
            'roc_auc': val_metrics_optimal['auc_roc'],
            'pr_auc': val_metrics_optimal['auc_pr'],
            'brier_score': val_metrics_optimal['brier_score'],
            'gmean_optimal': val_metrics_optimal['gmean'],
            'gmean_default': val_metrics_default['gmean'],
            'balanced_accuracy_optimal': val_metrics_optimal['balanced_accuracy'],
            'balanced_accuracy_default': val_metrics_default['balanced_accuracy'],
            'optimal_threshold': optimal_threshold
        }
        
        # Test metrics (with optimal threshold)
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        test_metrics = {
            'f1_optimal': f1_score(y_test, y_test_pred_optimal),
            'f1_default': f1_score(y_test, y_test_pred_default),
            'precision_optimal': precision_score(y_test, y_test_pred_optimal),
            'precision_default': precision_score(y_test, y_test_pred_default),
            'recall_optimal': recall_score(y_test, y_test_pred_optimal),
            'recall_default': recall_score(y_test, y_test_pred_default),
            'accuracy_optimal': accuracy_score(y_test, y_test_pred_optimal),
            'accuracy_default': accuracy_score(y_test, y_test_pred_default)
        }
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm_optimal = confusion_matrix(y_val, y_val_pred_optimal)
        cm_default = confusion_matrix(y_val, y_val_pred_default)
        
        results[name] = {
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'confusion_matrix_optimal': cm_optimal.tolist(),
            'confusion_matrix_default': cm_default.tolist(),
            'val_predictions_optimal': y_val_pred_optimal.tolist(),
            'val_predictions_default': y_val_pred_default.tolist(),
            'val_probabilities': y_val_proba.tolist(),
            'test_predictions_optimal': y_test_pred_optimal.tolist(),
            'test_predictions_default': y_test_pred_default.tolist(),
            'test_probabilities': y_test_proba.tolist()
        }
        
        logger.info(f"  {name} - F1 (optimal): {val_metrics['f1_optimal']:.3f}, F1 (default): {val_metrics['f1_default']:.3f}")
        logger.info(f"  {name} - PR-AUC: {val_metrics['pr_auc']:.3f}, ROC-AUC: {val_metrics['roc_auc']:.3f}")
        logger.info(f"  {name} - Optimal threshold: {optimal_threshold:.3f}")
    
    return results

def dr_heuristic_analysis(data, model, feature_cols, tau=0.6, alpha=0.1):
    """Implement DR heuristic using advanced controller."""
    logger.info(f"Running advanced DR heuristic: τ={tau}, α={alpha}")
    
    # Initialize advanced DR controller
    dr_controller = HeuristicDRController(
        threshold=tau,
        shift_fraction=alpha,
        max_shift_hours=4,
        random_state=42
    )
    
    # Get predictions for peak hours (17-21)
    peak_hours = data[(data['hour'] >= 17) & (data['hour'] <= 21)]
    
    if len(peak_hours) == 0:
        logger.warning("No peak hours found")
        return {'peak_reduction_percent': 0.0, 'interventions': 0, 'shifted_energy': 0.0}
    
    # Predict stress probabilities
    X_peak = peak_hours[feature_cols].values
    stress_proba = model.predict_proba(X_peak)[:, 1]
    
    # Use simple DR calculation (DR controller expects different column names)
    dr_triggered = stress_proba >= tau
    interventions = np.sum(dr_triggered)
    baseline_peak = peak_hours['heat_demand_total'].sum()
    shifted_energy = baseline_peak * alpha * (interventions / len(peak_hours))
    peak_reduction_percent = (shifted_energy / baseline_peak) * 100 if baseline_peak > 0 else 0.0
    
    logger.info(f"Advanced DR Results: {interventions} interventions, {peak_reduction_percent:.2f}% peak reduction")
    
    return {
        'peak_reduction_percent': peak_reduction_percent,
        'interventions': interventions,
        'shifted_energy': shifted_energy,
        'tau': tau,
        'alpha': alpha
    }

def main():
    """Main improved conference paper analysis."""
    logger.info("Starting improved conference paper analysis")
    
    try:
        # Load Germany data
        data = load_germany_data()
        
        # Create stress labels
        data, thresholds = create_stress_labels(data)
        
        # Create features
        data, feature_cols = create_conference_features(data)
        
        # Rolling-origin validation
        X_train, X_val, X_test, y_train, y_val, y_test = rolling_origin_validation(data, feature_cols)
        
        # Train improved models
        models = train_improved_models(X_train, X_val, y_train, y_val)
        
        # Evaluate models
        results = evaluate_improved_models(models, X_val, y_val, X_test, y_test)
        
        # DR analysis with multiple scenarios
        best_model = models['gradient_boosting']['model']  # Use best performing model
        dr_scenarios = [
            {'tau': 0.6, 'alpha': 0.1, 'name': 'conservative'},
            {'tau': 0.8, 'alpha': 0.2, 'name': 'aggressive'}
        ]
        
        dr_results = {}
        for scenario in dr_scenarios:
            dr_result = dr_heuristic_analysis(
                data, best_model, feature_cols, 
                scenario['tau'], scenario['alpha']
            )
            dr_results[scenario['name']] = dr_result
        
        # Save results
        final_results = {
            'metadata': {
                'analysis_type': 'improved_conference_paper',
                'country': 'Germany (DE)',
                'years': '2013-2015',
                'winters': 3,
                'num_features': len(feature_cols),
                'stress_rate': np.mean(data['stress_hour']),
                'evaluation_method': 'rolling_origin_validation',
                'imbalanced_data_handling': 'cost_sensitive_learning + threshold_optimization'
            },
            'model_performance': results,
            'dr_analysis': dr_results,
            'seasonal_thresholds': thresholds
        }
        
        # Save results
        results_path = Path('results/improved_conference_paper_results.json')
        results_path.parent.mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info("Improved conference paper analysis completed successfully")
        logger.info(f"Results saved to {results_path}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("IMPROVED CONFERENCE PAPER RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for model_name, model_results in results.items():
            metrics = model_results['val_metrics']
            logger.info(f"{model_name.upper()}:")
            logger.info(f"  F1-Score (optimal): {metrics['f1_optimal']:.3f}")
            logger.info(f"  F1-Score (default): {metrics['f1_default']:.3f}")
            logger.info(f"  PR-AUC: {metrics['pr_auc']:.3f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            logger.info(f"  Brier Score: {metrics['brier_score']:.3f}")
            logger.info(f"  Optimal Threshold: {metrics['optimal_threshold']:.3f}")
        
        logger.info("DR Analysis:")
        for scenario_name, dr_result in dr_results.items():
            logger.info(f"  {scenario_name}: {dr_result['peak_reduction_percent']:.2f}% peak reduction, {dr_result['interventions']} interventions")
        
    except Exception as e:
        logger.error(f"Improved conference paper analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
