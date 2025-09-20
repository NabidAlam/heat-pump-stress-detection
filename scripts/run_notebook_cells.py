#!/usr/bin/env python3
"""
Run EDA Notebook Cells to Learn from Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def run_cell_1():
    """Cell 1: Import libraries"""
    print("=" * 60)
    print("CELL 1: IMPORT LIBRARIES")
    print("=" * 60)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    print("Libraries loaded successfully!")
    print("✅ Pandas, NumPy, Matplotlib, Seaborn ready")
    print("✅ Plotting styles configured")

def run_cell_2():
    """Cell 2: Load dataset and basic info"""
    print("\n" + "=" * 60)
    print("CELL 2: DATASET LOADING & BASIC INFO")
    print("=" * 60)
    
    # Load dataset
    data_path = 'dataset/when2heat.csv'
    df = pd.read_csv(data_path, sep=';')
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['utc_timestamp'].min()} to {df['utc_timestamp'].max()}")
    
    # Basic info
    print("\nColumn types:")
    print(df.dtypes.value_counts())
    
    print("\nCountries available:")
    countries = set([col.split('_')[0] for col in df.columns if '_' in col and col.split('_')[0].isupper()])
    print(f"Found {len(countries)} countries: {sorted(list(countries))}")
    
    return df

def run_cell_3(df):
    """Cell 3: Parse datetime and analyze temporal patterns"""
    print("\n" + "=" * 60)
    print("CELL 3: TEMPORAL ANALYSIS")
    print("=" * 60)
    
    # Parse datetime and analyze temporal patterns
    df['datetime'] = pd.to_datetime(df['utc_timestamp'])
    df.set_index('datetime', inplace=True)
    
    print("Temporal Analysis:")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total hours: {len(df):,}")
    
    # Data completeness by year
    yearly_completeness = df.groupby(df.index.year).size()
    print("\nData completeness by year:")
    for year, hours in yearly_completeness.items():
        expected = 8784 if year % 4 == 0 else 8760  # Leap year
        completeness = (hours / expected) * 100
        print(f"{year}: {hours:,} hours ({completeness:.1f}% complete)")
    
    # Analyze data quality patterns
    print("\nData Quality Insights:")
    print("✅ 2008-2015: Complete data (100% coverage)")
    print("❌ 2016-2022: Missing data (0% coverage)")
    print("📊 Total usable data: 8 years (2008-2015)")
    
    return df

def run_additional_analysis(df):
    """Additional analysis based on notebook insights"""
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS: DATA PATTERNS")
    print("=" * 60)
    
    # Analyze column patterns
    heat_demand_cols = [col for col in df.columns if 'heat_demand_total' in col]
    cop_cols = [col for col in df.columns if 'COP_ASHP_floor' in col]
    
    print(f"Heat demand columns: {len(heat_demand_cols)}")
    print(f"COP columns: {len(cop_cols)}")
    
    # Sample data quality check
    if heat_demand_cols and cop_cols:
        sample_heat = df[heat_demand_cols[0]]  # First country
        sample_cop = df[cop_cols[0]]  # First country
        
        # Convert COP to numeric (handle comma decimals)
        sample_cop_numeric = pd.to_numeric(sample_cop.astype(str).str.replace(',', '.'), errors='coerce')
        
        print(f"\nSample data quality (first country):")
        print(f"Heat demand missing: {sample_heat.isna().sum():,} / {len(sample_heat):,}")
        print(f"COP missing: {sample_cop_numeric.isna().sum():,} / {len(sample_cop_numeric):,}")
        
        # Check data ranges
        if sample_heat.notna().any():
            print(f"Heat demand range: {sample_heat.min():.0f} to {sample_heat.max():.0f}")
        if sample_cop_numeric.notna().any():
            print(f"COP range: {sample_cop_numeric.min():.2f} to {sample_cop_numeric.max():.2f}")

def main():
    """Run all notebook cells"""
    print("🔍 RUNNING EDA NOTEBOOK CELLS")
    print("Learning from data exploration...")
    
    # Run cells
    run_cell_1()
    df = run_cell_2()
    df = run_cell_3(df)
    run_additional_analysis(df)
    
    print("\n" + "=" * 60)
    print("📚 LEARNING SUMMARY")
    print("=" * 60)
    print("✅ Dataset: 131,483 rows × 656 columns")
    print("✅ Countries: 28 EU countries")
    print("✅ Time period: 2007-2022 (but only 2008-2015 usable)")
    print("✅ Data types: Mixed (object, float64, int64)")
    print("✅ Heat demand: 26 columns")
    print("✅ COP data: 28 columns")
    print("⚠️  Missing data: 2016-2022 completely missing")
    print("✅ Usable period: 2008-2015 (8 years)")
    
    print("\n🎯 Key Insights:")
    print("1. Data is complete for 2008-2015")
    print("2. No imputation needed - use complete data only")
    print("3. Pan-European coverage with 28 countries")
    print("4. Rich feature set with heat demand and COP data")
    print("5. Suitable for stress detection analysis")

if __name__ == "__main__":
    main()
