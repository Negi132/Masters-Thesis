"""
Recalculate Naive Baseline and Fix NaN MAE Values
==================================================
The Naive_Persistence rows exist but their MAE values are NaN.
This script recalculates the naive baseline and updates the CSV.

Run from /Code/ML_Pipeline/ directory.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Make sure we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    MASTER_PATH = config.ML_DATA_DIR / "Master_Matrix_DK1_Horizon0h.csv"
    print(f"Using config.ML_DATA_DIR: {config.ML_DATA_DIR}")
except ImportError:
    # Fallback: hardcode the path
    print("Could not import config, using hardcoded path")
    MASTER_PATH = Path("../Data_Engineering/Data/ML_Ready_Data/Master_Matrix_DK1_Horizon0h.csv")

RAW_CSV = "experiment_results.csv"

print("=" * 80)
print("RECALCULATING NAIVE BASELINE")
print("=" * 80)

print(f"\nLooking for master matrix at: {MASTER_PATH}")
print(f"Absolute path: {MASTER_PATH.resolve() if hasattr(MASTER_PATH, 'resolve') else os.path.abspath(MASTER_PATH)}")

if not os.path.exists(MASTER_PATH):
    print(f"\nERROR: Could not find file at {MASTER_PATH}")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

print(f"\nLoading master matrix...")
df_master = pd.read_csv(MASTER_PATH)
print(f"  Loaded {len(df_master)} rows")
print(f"  First 5 columns: {list(df_master.columns)[:5]}")

if 'SpotPriceEUR' not in df_master.columns:
    print(f"ERROR: SpotPriceEUR column not found!")
    print(f"Available columns: {list(df_master.columns)}")
    sys.exit(1)

# Calculate naive MAE for each target/horizon combination
print(f"\nCalculating naive persistence baselines...")
print(f"  {'Target':<10} {'Horizon':<10} {'Naive MAE':>15}  {'N samples':>10}")
print(f"  {'-'*55}")

naive_results = []
horizons = [0, 24, 48, 72, 96, 120, 144, 168]

for target in ["Price", "Delta"]:
    if target == "Delta":
        df_master['Delta_Actual'] = df_master['SpotPriceEUR'] - df_master['SpotPriceEUR'].shift(24)
    
    for h in horizons:
        if target == "Delta" and h == 0:
            continue
        
        if target == "Price":
            pred = df_master['SpotPriceEUR'].shift(h)
            actual = df_master['SpotPriceEUR']
        else:
            prev_delta = df_master['SpotPriceEUR'] - df_master['SpotPriceEUR'].shift(24)
            pred = prev_delta.shift(h)
            actual = df_master['Delta_Actual']
        
        valid_idx = pred.notna() & actual.notna()
        n_valid = valid_idx.sum()
        
        if n_valid == 0:
            print(f"  {target:<10} {h:>3}h       {'N/A':>15}  {0:>10}")
            continue
        
        mae = np.mean(np.abs(actual[valid_idx] - pred[valid_idx]))
        print(f"  {target:<10} {h:>3}h       {mae:>15.4f}  {n_valid:>10}")
        
        naive_results.append({
            'Target': target,
            'Horizon': h,
            'MAE': round(mae, 4)
        })

print(f"\nCalculated {len(naive_results)} naive baselines")

# Update the CSV
print("\n" + "=" * 80)
print("UPDATING CSV")
print("=" * 80)

print(f"\nLoading raw CSV: {RAW_CSV}")
df_csv = pd.read_csv(RAW_CSV, sep=None, engine='python')
print(f"  Total rows: {len(df_csv)}")

# Find existing Naive rows
naive_mask = df_csv['Model'] == 'Naive_Persistence'
existing_naive = naive_mask.sum()
print(f"  Existing Naive_Persistence rows: {existing_naive}")

if existing_naive > 0:
    print(f"\nUpdating existing rows...")
    updated = 0
    for result in naive_results:
        target = result['Target']
        h = result['Horizon']
        target_col = f"TARGET_{target}_{h}h"
        
        match = (df_csv['Model'] == 'Naive_Persistence') & (df_csv['Target'] == target_col)
        if match.sum() > 0:
            df_csv.loc[match, 'MAE'] = result['MAE']
            df_csv.loc[match, 'Status'] = 'SUCCESS'
            updated += match.sum()
    print(f"  Updated {updated} rows with proper MAE values")
else:
    print(f"\nAdding new rows...")
    new_rows = []
    for result in naive_results:
        target = result['Target']
        h = result['Horizon']
        new_rows.append({
            'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Experiment': f"Naive_Baseline_Persistence_{target}_{h}h",
            'Region': 'DK1',
            'Target': f"TARGET_{target}_{h}h",
            'Feature_Mask': '[]',
            'Status': 'SUCCESS',
            'RMSE': np.nan,
            'MAE': result['MAE'],
            'R2': np.nan,
            'WMAPE': np.nan,
            'sMAPE': np.nan,
            'MDA': np.nan,
            'Train_Time_Sec': 0.0,
            'Model': 'Naive_Persistence'
        })
    df_new = pd.DataFrame(new_rows)
    df_csv = pd.concat([df_csv, df_new], ignore_index=True)
    print(f"  Added {len(new_rows)} new rows")

# Save
df_csv.to_csv(RAW_CSV, index=False)
print(f"\nSaved updated CSV: {RAW_CSV}")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("1. python cleanDuplicateCSV.py")
print("2. cd ../Plotting && python master_plotter.py")
print("=" * 80)
