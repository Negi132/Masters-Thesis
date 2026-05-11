"""
=============================================================================
WARNING: MASTER FEATURE MATRIX - DO NOT FEED DIRECTLY TO ML MODELS
=============================================================================
This script generates the foundational "Master Data Matrix" containing 
all engineered features, historical lags, cyclical time encodings, and targets.

CRITICAL MLOPS DEPENDENCY:
This dataset contains explicit Data Leakage by design. 
The raw `SpotPriceEUR` and concurrent 0-hour deltas are preserved in this 
file to act as base references for future calculations. 

If you feed this CSV directly into a model like XGBoost or a Neural Network, 
the model will cheat, achieve 100% accuracy, and fail in the real world.

This file MUST be routed through the `Dynamic Masking DataLoader`, which 
enforces a strict "Target Shield" to drop all unused targets and raw 
leaky variables from RAM before training begins.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")
ALIGNED_DIR = BASE_PATH / "Aligned_Yearly"
DMI_ALIGNED_DIR = BASE_PATH / "DMI" / "AlignedZones"
OUTPUT_DIR = BASE_PATH / "ML_Ready_Data"

# The column we want to predict
TARGET_COLUMN = "SpotPriceEUR"

# 0 = Predict the current hour (Oracle Baseline). 
TARGET_HORIZON = 0 

# Historical market memory for the Price
PRICE_LAGS = [24, 48, 168]

# Universal lag applied to ALL weather and grid features
UNIVERSAL_LAG = 24 

def engineer_time_features(df):
    """Creates cyclical time features for Neural Networks."""
    df['hour'] = df['HourUTC_dt'].dt.hour
    df['month'] = df['HourUTC_dt'].dt.month
    df['dayofweek'] = df['HourUTC_dt'].dt.dayofweek
    df['dayofyear'] = df['HourUTC_dt'].dt.dayofyear
    
    # Sine/Cosine transformations
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    return df

def build_region_master(region):
    print(f"\nBuilding Master Matrix for {region}...")
    
    all_merged_years = []
    
    # 1. Load and merge all years for this specific region
    # FIX: Added "_hourly" to the glob search string
    years = sorted(list(set([f.name.split('_')[0] for f in DMI_ALIGNED_DIR.glob(f"*_{region}_hourly_aligned.csv")])))
    
    for year in years:
        identifier = f"{year}_{region}"
        
        # FIX: Added "_hourly" to the explicit DMI path
        dmi_path = DMI_ALIGNED_DIR / f"{identifier}_hourly_aligned.csv"
        price_path = ALIGNED_DIR / f"Prices_{identifier}.csv"
        prod_path = ALIGNED_DIR / f"ProdCons_{identifier}.csv"
        
        if not (dmi_path.exists() and price_path.exists() and prod_path.exists()):
            continue
            
        df_dmi = pd.read_csv(dmi_path)
        df_price = pd.read_csv(price_path)
        df_prod = pd.read_csv(prod_path)
        
        # Merge the three domains for this year
        df_year = pd.merge(df_price, df_prod, on=['HourUTC', 'PriceArea'], how='inner')
        df_year = pd.merge(df_year, df_dmi, on='HourUTC', how='inner')
        
        all_merged_years.append(df_year)
        
    if not all_merged_years:
        print(f"  [ERROR] No complete data found for {region}.")
        return

    # 2. Stack the 10 years into one massive timeline
    master_df = pd.concat(all_merged_years, ignore_index=True)
    master_df['HourUTC_dt'] = pd.to_datetime(master_df['HourUTC'], utc=True)
    master_df = master_df.sort_values('HourUTC_dt').reset_index(drop=True)
    
    print(f"  -> Merged {len(years)} years: {len(master_df)} total hours.")

    # 3. Feature Engineering (Time)
    print("  -> Engineering Time Encodings...")
    master_df = engineer_time_features(master_df)
    
    # 4. Feature Engineering (Universal Lags)
    time_cols = ['hour', 'month', 'dayofweek', 'dayofyear', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos']
    numeric_cols = master_df.select_dtypes(include=['number']).columns
    cols_to_lag = [c for c in numeric_cols if c not in time_cols and c != TARGET_COLUMN]

    print(f"  -> Generating {UNIVERSAL_LAG}h lags for {len(cols_to_lag)} physical features...")
    for col in cols_to_lag:
        master_df[f'{col}_lag_{UNIVERSAL_LAG}h'] = master_df[col].shift(UNIVERSAL_LAG)

    # 5. Feature Engineering (Deep Price Lags & Deltas)
    print("  -> Generating deep historical lags and deltas for Spot Price...")
    for lag in PRICE_LAGS:
        lag_col = f'SpotPriceEUR_lag_{lag}h'
        if lag_col not in master_df.columns:
            master_df[lag_col] = master_df[TARGET_COLUMN].shift(lag)
            
        # The Feature Delta (Current Price minus Past Price)
        delta_col = f'SpotPriceEUR_historical_delta_{lag}h'
        master_df[delta_col] = master_df[TARGET_COLUMN] - master_df[lag_col]

    # 6. Target Shifting (The Answer Key)
    print(f"  -> Generating Target Variables (Horizon: +{TARGET_HORIZON}h)...")
    target_name = f'TARGET_Price_{TARGET_HORIZON}h'
    target_delta_name = f'TARGET_Delta_{TARGET_HORIZON}h'
    
    if TARGET_HORIZON == 0:
        master_df[target_name] = master_df[TARGET_COLUMN] 
        master_df[target_delta_name] = 0.0
    else:
        master_df[target_name] = master_df[TARGET_COLUMN].shift(-TARGET_HORIZON)
        master_df[target_delta_name] = master_df[target_name] - master_df[TARGET_COLUMN]

    # 7. Handle NaNs Intelligently
    initial_rows = len(master_df)
    
    # A. Only drop rows where the absolute edges were created by our specific shifts
    edge_columns = [target_name] + [f'SpotPriceEUR_lag_{lag}h' for lag in PRICE_LAGS]
    master_df = master_df.dropna(subset=edge_columns)
    
    # B. Fill any remaining historical NaNs (e.g., market types or solar farms that didn't exist in 2015) with 0
    master_df = master_df.fillna(0)
    
    print(f"  -> Dropped {initial_rows - len(master_df)} edge rows due to NaN shifts.")

    # Clean up intermediate datetime column
    master_df = master_df.drop(columns=['HourUTC_dt'])

    # 8. Save Final Matrix
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"Master_Matrix_{region}_Horizon{TARGET_HORIZON}h.csv"
    master_df.to_csv(out_file, index=False)
    
    print(f"  [SUCCESS] Saved ML-Ready Dataset to {out_file.name}")
    print(f"  Final Shape: {master_df.shape[0]} rows, {master_df.shape[1]} features.")

def main():
    print("==================================================")
    print("      MASTER ML FEATURE GENERATOR")
    print("==================================================")
    build_region_master("DK1")
    build_region_master("DK2")
    print("==================================================")
    print("All ML Data ready! Next step: The Walk-Forward Trainer.")

if __name__ == "__main__":
    main()