#!/usr/bin/env python3
"""
Heuristic Demand Response Controller
====================================

This module provides heuristic-based demand response control strategies
for energy system stress mitigation.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class HeuristicDRController:
    """
    Heuristic-based demand response controller.
    
    Implements various heuristic strategies for demand response
    based on system conditions and stress indicators.
    """
    
    def __init__(self, 
                 threshold: float = 0.6,
                 shift_fraction: float = 0.1,
                 max_shift_hours: int = 4,
                 random_state: int = 42):
        """
        Initialize the heuristic DR controller.
        
        Args:
            threshold: Threshold for triggering DR actions
            shift_fraction: Fraction of demand that can be shifted
            max_shift_hours: Maximum hours for demand shifting
            random_state: Random state for reproducibility
        """
        self.threshold = threshold
        self.shift_fraction = shift_fraction
        self.max_shift_hours = max_shift_hours
        self.random_state = random_state
        
        logger.info(f"Initialized HeuristicDRController with threshold={threshold}")
    
    def apply_dr_strategy(self, 
                         data: pd.DataFrame, 
                         stress_hours: pd.Series,
                         strategy: str = 'conservative') -> Dict[str, Any]:
        """
        Apply demand response strategy to reduce stress.
        
        Args:
            data: Energy system data
            stress_hours: Boolean series indicating stress hours
            strategy: DR strategy to apply ('conservative', 'aggressive', 'adaptive')
            
        Returns:
            Dictionary with DR results
        """
        logger.info(f"Applying {strategy} DR strategy")
        
        # Create a copy for simulation
        sim_data = data.copy()
        
        # Identify stress hours
        stress_indices = stress_hours[stress_hours].index
        
        if len(stress_indices) == 0:
            logger.warning("No stress hours found for DR application")
            return self._create_empty_results()
        
        # Apply strategy-specific DR actions
        if strategy == 'conservative':
            dr_actions = self._apply_conservative_strategy(sim_data, stress_indices)
        elif strategy == 'aggressive':
            dr_actions = self._apply_aggressive_strategy(sim_data, stress_indices)
        elif strategy == 'adaptive':
            dr_actions = self._apply_adaptive_strategy(sim_data, stress_indices)
        else:
            raise ValueError(f"Unknown DR strategy: {strategy}")
        
        # Calculate results
        results = self._calculate_dr_results(
            data, sim_data, stress_hours, dr_actions, strategy
        )
        
        logger.info(f"{strategy} DR strategy completed")
        return results
    
    def _apply_conservative_strategy(self, 
                                   data: pd.DataFrame, 
                                   stress_indices: pd.Index) -> List[Dict[str, Any]]:
        """Apply conservative DR strategy."""
        dr_actions = []
        
        for stress_idx in stress_indices:
            # Conservative: small demand reduction
            current_demand = data.loc[stress_idx, 'heat_demand_total']
            reduction = current_demand * (self.shift_fraction * 0.5)  # 50% of normal shift
            
            # Apply reduction
            data.loc[stress_idx, 'heat_demand_total'] -= reduction
            
            # Record action
            action = {
                'stress_hour': stress_idx,
                'action_type': 'demand_reduction',
                'demand_change': -reduction,
                'strategy': 'conservative'
            }
            dr_actions.append(action)
        
        return dr_actions
    
    def _apply_aggressive_strategy(self, 
                                 data: pd.DataFrame, 
                                 stress_indices: pd.Index) -> List[Dict[str, Any]]:
        """Apply aggressive DR strategy."""
        dr_actions = []
        
        for stress_idx in stress_indices:
            # Aggressive: larger demand reduction
            current_demand = data.loc[stress_idx, 'heat_demand_total']
            reduction = current_demand * (self.shift_fraction * 1.5)  # 150% of normal shift
            
            # Apply reduction
            data.loc[stress_idx, 'heat_demand_total'] -= reduction
            
            # Record action
            action = {
                'stress_hour': stress_idx,
                'action_type': 'demand_reduction',
                'demand_change': -reduction,
                'strategy': 'aggressive'
            }
            dr_actions.append(action)
        
        return dr_actions
    
    def _apply_adaptive_strategy(self, 
                               data: pd.DataFrame, 
                               stress_indices: pd.Index) -> List[Dict[str, Any]]:
        """Apply adaptive DR strategy."""
        dr_actions = []
        
        for stress_idx in stress_indices:
            # Adaptive: adjust based on stress severity
            current_demand = data.loc[stress_idx, 'heat_demand_total']
            current_cop = data.loc[stress_idx, 'cop_average']
            
            # Calculate stress severity
            demand_percentile = data['heat_demand_total'].rolling(168).quantile(0.9).loc[stress_idx]
            cop_percentile = data['cop_average'].rolling(168).quantile(0.1).loc[stress_idx]
            
            demand_severity = current_demand / demand_percentile
            cop_severity = cop_percentile / current_cop
            
            # Adaptive reduction based on severity
            severity_factor = (demand_severity + cop_severity) / 2
            reduction = current_demand * self.shift_fraction * min(severity_factor, 2.0)
            
            # Apply reduction
            data.loc[stress_idx, 'heat_demand_total'] -= reduction
            
            # Record action
            action = {
                'stress_hour': stress_idx,
                'action_type': 'adaptive_reduction',
                'demand_change': -reduction,
                'severity_factor': severity_factor,
                'strategy': 'adaptive'
            }
            dr_actions.append(action)
        
        return dr_actions
    
    def _calculate_dr_results(self, 
                            original_data: pd.DataFrame,
                            sim_data: pd.DataFrame,
                            stress_hours: pd.Series,
                            dr_actions: List[Dict[str, Any]],
                            strategy: str) -> Dict[str, Any]:
        """Calculate DR results and impact metrics."""
        # Recalculate stress hours after DR
        stress_hours_after = self._recalculate_stress_hours(sim_data)
        
        # Calculate metrics
        original_stress_count = stress_hours.sum()
        after_stress_count = stress_hours_after.sum()
        
        stress_reduction = original_stress_count - after_stress_count
        stress_reduction_percentage = (stress_reduction / original_stress_count * 100) if original_stress_count > 0 else 0
        
        # Calculate demand impact
        total_demand_change = sum(action['demand_change'] for action in dr_actions)
        
        return {
            'strategy': strategy,
            'dr_actions': dr_actions,
            'impact_metrics': {
                'stress_hours_original': int(original_stress_count),
                'stress_hours_after_dr': int(after_stress_count),
                'stress_reduction': int(stress_reduction),
                'stress_reduction_percentage': stress_reduction_percentage,
                'total_demand_change': total_demand_change,
                'dr_actions_count': len(dr_actions)
            },
            'strategy_parameters': {
                'threshold': self.threshold,
                'shift_fraction': self.shift_fraction,
                'max_shift_hours': self.max_shift_hours
            }
        }
    
    def _recalculate_stress_hours(self, data: pd.DataFrame) -> pd.Series:
        """Recalculate stress hours after DR actions."""
        # Use the same logic as in preprocessing
        demand_percentile = 90.0
        cop_percentile = 10.0
        seasonal_window = 168
        
        # Calculate dynamic thresholds
        demand_threshold = data['heat_demand_total'].rolling(
            window=seasonal_window, center=False
        ).quantile(demand_percentile / 100)
        
        cop_threshold = data['cop_average'].rolling(
            window=seasonal_window, center=False
        ).quantile(cop_percentile / 100)
        
        # Detect stress conditions
        high_demand = data['heat_demand_total'] >= demand_threshold
        low_cop = data['cop_average'] <= cop_threshold
        
        return (high_demand & low_cop)
    
    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty results when no stress hours are found."""
        return {
            'strategy': 'none',
            'dr_actions': [],
            'impact_metrics': {
                'stress_hours_original': 0,
                'stress_hours_after_dr': 0,
                'stress_reduction': 0,
                'stress_reduction_percentage': 0,
                'total_demand_change': 0,
                'dr_actions_count': 0
            },
            'strategy_parameters': {
                'threshold': self.threshold,
                'shift_fraction': self.shift_fraction,
                'max_shift_hours': self.max_shift_hours
            }
        }
