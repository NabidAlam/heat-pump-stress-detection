#!/usr/bin/env python3
"""
Test Demand Response Heuristic
==============================

Tests for demand response heuristic functionality.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dr.heuristic import HeuristicDRController
from dr.simulation import DemandResponseSimulator


class TestDRHeuristic(unittest.TestCase):
    """Test demand response heuristic functionality."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create synthetic energy data
        self.data = pd.DataFrame({
            'heat_demand_total': np.random.exponential(100, n_samples),
            'cop_average': np.random.uniform(2, 5, n_samples),
            'cop_ashp': np.random.uniform(2, 5, n_samples)
        })
        self.data.index = pd.date_range('2020-01-01', periods=n_samples, freq='h')
        
        # Create stress hours (some hours with high demand and low COP)
        stress_mask = (
            (self.data['heat_demand_total'] > self.data['heat_demand_total'].quantile(0.9)) &
            (self.data['cop_average'] < self.data['cop_average'].quantile(0.1))
        )
        self.stress_hours = stress_mask
    
    def test_dr_controller_creation(self):
        """Test DR controller creation."""
        controller = HeuristicDRController(
            threshold=0.6,
            shift_fraction=0.1,
            max_shift_hours=4,
            random_state=42
        )
        
        self.assertEqual(controller.threshold, 0.6)
        self.assertEqual(controller.shift_fraction, 0.1)
        self.assertEqual(controller.max_shift_hours, 4)
    
    def test_dr_simulator_creation(self):
        """Test DR simulator creation."""
        simulator = DemandResponseSimulator(
            trigger_threshold=0.6,
            shift_fraction=0.1,
            max_shift_hours=4,
            random_state=42
        )
        
        self.assertEqual(simulator.trigger_threshold, 0.6)
        self.assertEqual(simulator.shift_fraction, 0.1)
        self.assertEqual(simulator.max_shift_hours, 4)
    
    def test_conservative_strategy(self):
        """Test conservative DR strategy."""
        controller = HeuristicDRController(
            threshold=0.6,
            shift_fraction=0.1,
            random_state=42
        )
        
        results = controller.apply_dr_strategy(
            self.data, self.stress_hours, strategy='conservative'
        )
        
        # Check results structure
        self.assertIn('strategy', results)
        self.assertIn('dr_actions', results)
        self.assertIn('impact_metrics', results)
        
        self.assertEqual(results['strategy'], 'conservative')
        self.assertIsInstance(results['dr_actions'], list)
        
        # Check impact metrics
        impact = results['impact_metrics']
        self.assertIn('stress_hours_original', impact)
        self.assertIn('stress_hours_after_dr', impact)
        self.assertIn('stress_reduction', impact)
    
    def test_aggressive_strategy(self):
        """Test aggressive DR strategy."""
        controller = HeuristicDRController(
            threshold=0.6,
            shift_fraction=0.1,
            random_state=42
        )
        
        results = controller.apply_dr_strategy(
            self.data, self.stress_hours, strategy='aggressive'
        )
        
        self.assertEqual(results['strategy'], 'aggressive')
        self.assertIsInstance(results['dr_actions'], list)
    
    def test_adaptive_strategy(self):
        """Test adaptive DR strategy."""
        controller = HeuristicDRController(
            threshold=0.6,
            shift_fraction=0.1,
            random_state=42
        )
        
        results = controller.apply_dr_strategy(
            self.data, self.stress_hours, strategy='adaptive'
        )
        
        self.assertEqual(results['strategy'], 'adaptive')
        self.assertIsInstance(results['dr_actions'], list)
    
    def test_dr_simulation(self):
        """Test DR simulation functionality."""
        simulator = DemandResponseSimulator(
            trigger_threshold=0.6,
            shift_fraction=0.1,
            random_state=42
        )
        
        results = simulator.simulate_dr_impact(self.data, self.stress_hours)
        
        # Check results structure
        self.assertIn('dr_actions', results)
        self.assertIn('impact_metrics', results)
        self.assertIn('simulation_parameters', results)
        
        # Check impact metrics
        impact = results['impact_metrics']
        self.assertIn('stress_hours_after_dr', impact)
        self.assertIn('stress_reduction', impact)
        self.assertIn('stress_reduction_percentage', impact)
    
    def test_no_stress_hours(self):
        """Test behavior when no stress hours are present."""
        # Create data with no stress hours
        no_stress_data = self.data.copy()
        no_stress_data['heat_demand_total'] = 50  # Low demand
        no_stress_data['cop_average'] = 4.5  # High COP
        
        no_stress_mask = pd.Series(False, index=no_stress_data.index)
        
        controller = HeuristicDRController(random_state=42)
        results = controller.apply_dr_strategy(
            no_stress_data, no_stress_mask, strategy='conservative'
        )
        
        # Should handle gracefully
        self.assertEqual(results['strategy'], 'none')
        self.assertEqual(len(results['dr_actions']), 0)
        self.assertEqual(results['impact_metrics']['stress_hours_original'], 0)
    
    def test_dr_impact_calculation(self):
        """Test DR impact calculation."""
        controller = HeuristicDRController(
            shift_fraction=0.2,  # 20% shift
            random_state=42
        )
        
        results = controller.apply_dr_strategy(
            self.data, self.stress_hours, strategy='conservative'
        )
        
        # Check that stress reduction is calculated
        impact = results['impact_metrics']
        self.assertGreaterEqual(impact['stress_reduction'], 0)
        self.assertGreaterEqual(impact['stress_reduction_percentage'], 0)
        
        # Check that total demand change is calculated
        self.assertIn('total_demand_change', impact)


if __name__ == '__main__':
    unittest.main()
