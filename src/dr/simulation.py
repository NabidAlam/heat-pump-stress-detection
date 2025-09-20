#!/usr/bin/env python3
"""
Demand Response Simulation
==========================

This module provides demand response simulation capabilities for
analyzing the impact of DR strategies on energy system stress.
"""

import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DemandResponseSimulator:
    """
    Simulate demand response strategies for energy system stress mitigation.
    
    This class provides methods to simulate various DR strategies and
    analyze their impact on system stress levels.
    """
    
    def __init__(self, 
                 trigger_threshold: float = 0.6,
                 shift_fraction: float = 0.1,
                 max_shift_hours: int = 4,
                 random_state: int = 42):
        """
        Initialize the DR simulator.
        
        Args:
            trigger_threshold: Threshold for triggering DR actions
            shift_fraction: Fraction of demand that can be shifted
            max_shift_hours: Maximum hours for demand shifting
            random_state: Random state for reproducibility
        """
        self.trigger_threshold = trigger_threshold
        self.shift_fraction = shift_fraction
        self.max_shift_hours = max_shift_hours
        self.random_state = random_state
        
        logger.info(f"Initialized DR Simulator with threshold={trigger_threshold}, "
                   f"shift_fraction={shift_fraction}")
    
    def simulate_dr_impact(self, 
                          data: pd.DataFrame, 
                          stress_hours: pd.Series) -> Dict[str, Any]:
        """
        Simulate the impact of demand response on system stress.
        
        Args:
            data: Energy system data
            stress_hours: Boolean series indicating stress hours
        
        Returns:
            Dictionary with DR simulation results
        """
        logger.info("Starting DR impact simulation")
        
        # Create a copy of the data for simulation
        sim_data = data.copy()
        
        # Identify stress hours
        stress_indices = stress_hours[stress_hours].index
        
        if len(stress_indices) == 0:
            logger.warning("No stress hours found for DR simulation")
            return self._create_empty_results()
        
        logger.info(f"Found {len(stress_indices)} stress hours for DR simulation")
        
        # Simulate DR actions
        dr_actions = self._simulate_dr_actions(sim_data, stress_indices)
        
        # Calculate impact metrics
        impact_metrics = self._calculate_impact_metrics(
            data, sim_data, stress_hours, dr_actions
        )
        
        # Create results
        results = {
            'dr_actions': dr_actions,
            'impact_metrics': impact_metrics,
            'simulation_parameters': {
                'trigger_threshold': self.trigger_threshold,
                'shift_fraction': self.shift_fraction,
                'max_shift_hours': self.max_shift_hours
            },
            'stress_hours_original': len(stress_indices),
            'stress_hours_after_dr': impact_metrics.get('stress_hours_after_dr', 0)
        }
        
        logger.info("DR impact simulation completed")
        return results
    
    def _simulate_dr_actions(self, 
                           data: pd.DataFrame, 
                           stress_indices: pd.Index) -> pd.DataFrame:
        """Simulate DR actions for stress hours."""
        dr_actions = []
        
        for stress_idx in stress_indices:
            # Get current demand
            current_demand = data.loc[stress_idx, 'heat_demand_total']
            
            # Calculate shiftable demand
            shiftable_demand = current_demand * self.shift_fraction
            
            # Find optimal shift time (within max_shift_hours)
            shift_time = self._find_optimal_shift_time(
                data, stress_idx, shiftable_demand
            )
            
            if shift_time is not None:
                # Record DR action
                action = {
                    'stress_hour': stress_idx,
                    'original_demand': current_demand,
                    'shiftable_demand': shiftable_demand,
                    'shift_time': shift_time,
                    'action_type': 'demand_shift'
                }
                dr_actions.append(action)
                
                # Apply DR action
                data.loc[stress_idx, 'heat_demand_total'] -= shiftable_demand
                data.loc[shift_time, 'heat_demand_total'] += shiftable_demand
        
        return pd.DataFrame(dr_actions)
    
    def _find_optimal_shift_time(self, 
                               data: pd.DataFrame, 
                               stress_idx: pd.Index, 
                               shiftable_demand: float) -> Optional[pd.Index]:
        """Find optimal time to shift demand."""
        # Look for times within max_shift_hours where demand is lower
        time_range = pd.date_range(
            start=stress_idx - pd.Timedelta(hours=self.max_shift_hours),
            end=stress_idx + pd.Timedelta(hours=self.max_shift_hours),
            freq='h'
        )
        
        # Filter to available times in data
        available_times = time_range.intersection(data.index)
        
        if len(available_times) == 0:
            return None
        
        # Find time with lowest demand
        demands = data.loc[available_times, 'heat_demand_total']
        optimal_time = demands.idxmin()
        
        # Check if shift is beneficial
        if demands.loc[optimal_time] < data.loc[stress_idx, 'heat_demand_total']:
            return optimal_time
        
        return None
    
    def _calculate_impact_metrics(self, 
                                original_data: pd.DataFrame,
                                sim_data: pd.DataFrame,
                                stress_hours: pd.Series,
                                dr_actions: pd.DataFrame) -> Dict[str, Any]:
        """Calculate impact metrics of DR simulation."""
        # Recalculate stress hours after DR
        stress_hours_after = self._recalculate_stress_hours(sim_data)
        
        # Calculate metrics
        original_stress_count = stress_hours.sum()
        after_stress_count = stress_hours_after.sum()
        
        stress_reduction = original_stress_count - after_stress_count
        stress_reduction_percentage = (stress_reduction / original_stress_count * 100) if original_stress_count > 0 else 0
        
        # Calculate demand impact
        original_total_demand = original_data['heat_demand_total'].sum()
        sim_total_demand = sim_data['heat_demand_total'].sum()
        demand_change = sim_total_demand - original_total_demand
        
        return {
            'stress_hours_after_dr': int(after_stress_count),
            'stress_reduction': int(stress_reduction),
            'stress_reduction_percentage': stress_reduction_percentage,
            'total_demand_change': demand_change,
            'dr_actions_count': len(dr_actions),
            'average_demand_shift': dr_actions['shiftable_demand'].mean() if len(dr_actions) > 0 else 0
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
            'dr_actions': pd.DataFrame(),
            'impact_metrics': {
                'stress_hours_after_dr': 0,
                'stress_reduction': 0,
                'stress_reduction_percentage': 0,
                'total_demand_change': 0,
                'dr_actions_count': 0,
                'average_demand_shift': 0
            },
            'simulation_parameters': {
                'trigger_threshold': self.trigger_threshold,
                'shift_fraction': self.shift_fraction,
                'max_shift_hours': self.max_shift_hours
            },
            'stress_hours_original': 0,
            'stress_hours_after_dr': 0
        }
