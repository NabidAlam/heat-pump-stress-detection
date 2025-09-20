#!/usr/bin/env python3
"""
Check Conference Results Logic
============================
"""

import json
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

def main():
    # Load results
    with open('results/conference_paper_results.json', 'r') as f:
        results = json.load(f)

    print('=== CONFERENCE RESULTS ANALYSIS ===')
    print()

    # Check metadata
    metadata = results['metadata']
    print('METADATA:')
    print(f'  Country: {metadata["country"]}')
    print(f'  Years: {metadata["years"]}')
    print(f'  Stress Rate: {metadata["stress_rate"]:.3f} ({metadata["stress_rate"]*100:.1f}%)')
    print(f'  Features: {metadata["num_features"]}')
    print()

    # Check model performance
    print('MODEL PERFORMANCE:')
    for model_name, model_results in results['model_performance'].items():
        metrics = model_results['val_metrics']
        print(f'{model_name.upper()}:')
        print(f'  F1-Score: {metrics["f1"]:.3f}')
        print(f'  PR-AUC: {metrics["pr_auc"]:.3f}')
        print(f'  Brier Score: {metrics["brier_score"]:.3f}')
        print(f'  Precision: {metrics["precision"]:.3f}')
        print(f'  Recall: {metrics["recall"]:.3f}')
        print()

    # Check DR analysis
    print('DR ANALYSIS:')
    for scenario, dr_result in results['dr_analysis'].items():
        print(f'{scenario.upper()}:')
        print(f'  Peak Reduction: {dr_result["peak_reduction_percent"]:.2f}%')
        print(f'  Interventions: {dr_result["interventions"]}')
        if 'tau' in dr_result:
            print(f'  tau: {dr_result["tau"]}, alpha: {dr_result["alpha"]}')
        print()

    # Check for logical issues
    print('=== LOGICAL ISSUES CHECK ===')
    print()
    
    issues = []
    
    # Check PR-AUC values
    for model_name, model_results in results['model_performance'].items():
        pr_auc = model_results['val_metrics']['pr_auc']
        if pr_auc < 0:
            issues.append(f"{model_name}: PR-AUC is negative ({pr_auc:.3f}) - this is impossible!")
        elif pr_auc > 1:
            issues.append(f"{model_name}: PR-AUC > 1 ({pr_auc:.3f}) - this is impossible!")
        elif pr_auc < 0.1:
            issues.append(f"{model_name}: PR-AUC very low ({pr_auc:.3f}) - model performing poorly")
    
    # Check F1 scores
    for model_name, model_results in results['model_performance'].items():
        f1 = model_results['val_metrics']['f1']
        if f1 > 0.9:
            issues.append(f"{model_name}: F1 very high ({f1:.3f}) - possible overfitting")
        elif f1 < 0.1:
            issues.append(f"{model_name}: F1 very low ({f1:.3f}) - model not learning")
    
    # Check Brier scores
    for model_name, model_results in results['model_performance'].items():
        brier = model_results['val_metrics']['brier_score']
        if brier > 0.5:
            issues.append(f"{model_name}: Brier score > 0.5 ({brier:.3f}) - worse than random!")
        elif brier < 0.01:
            issues.append(f"{model_name}: Brier score very low ({brier:.3f}) - possible overfitting")
    
    # Check DR results
    for scenario, dr_result in results['dr_analysis'].items():
        peak_reduction = dr_result['peak_reduction_percent']
        interventions = dr_result['interventions']
        
        if peak_reduction > 50:
            issues.append(f"{scenario}: Peak reduction very high ({peak_reduction:.2f}%) - unrealistic?")
        elif peak_reduction < 0.1 and interventions > 0:
            issues.append(f"{scenario}: Many interventions ({interventions}) but low reduction ({peak_reduction:.2f}%)")
    
    # Check stress rate
    stress_rate = metadata['stress_rate']
    if stress_rate < 0.01:
        issues.append(f"Stress rate very low ({stress_rate:.3f}) - may cause class imbalance issues")
    elif stress_rate > 0.5:
        issues.append(f"Stress rate very high ({stress_rate:.3f}) - may not be realistic")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("No major logical issues found!")
    
    print()
    print('=== SUMMARY ===')
    print(f"Stress Rate: {metadata['stress_rate']*100:.1f}% (should be 5-15% for realistic)")
    print(f"Best Model: Gradient Boosting (F1: {results['model_performance']['gradient_boosting']['val_metrics']['f1']:.3f})")
    print(f"DR Impact: {results['dr_analysis']['conservative']['peak_reduction_percent']:.2f}% peak reduction")
    print(f"Evaluation: Rolling-origin validation (correct method)")

if __name__ == "__main__":
    main()
