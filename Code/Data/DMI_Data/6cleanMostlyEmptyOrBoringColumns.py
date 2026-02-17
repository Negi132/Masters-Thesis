import pandas as pd
import glob
import os

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATTERN = "*_regional_timeseries.csv" 

# 1. Missing Threshold: Drop if > 90% empty
MISSING_THRESHOLD = 0.90 

# 2. Dominance Threshold: Drop if one single value appears in > 99% of rows
# (e.g., if 'snow_depth' is 0.0 for 99.5% of the year, drop it)
DOMINANCE_THRESHOLD = 0.95

# Safety List: NEVER delete these, even if they look constant or empty
ALWAYS_KEEP = [
    'Timestamp_UTC', 
    'wind_speed', 
    'radia_glob', 
    'temp_dry', 
    'mean_cloud_cover',
    'Stations_Reporting_Min' # Good to keep for QC
]

# ==========================================
# PROCESSING FUNCTION
# ==========================================

def clean_dataset(filename):
    print(f"Analyzing {filename}...")
    
    try:
        # Read CSV
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"  Error reading file: {e}")
        return

    rows = len(df)
    if rows == 0:
        print("  File is empty. Skipping.")
        return

    cols_to_drop = []
    reasons = {}

    # Iterate through columns
    for col in df.columns:
        if col in ALWAYS_KEEP:
            continue
            
        # -------------------------------------------------
        # CHECK 1: SPARSITY (Missing Values)
        # -------------------------------------------------
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / rows
        
        if missing_pct > MISSING_THRESHOLD:
            cols_to_drop.append(col)
            reasons[col] = f"Mostly Empty ({missing_pct:.1%})"
            continue # specific column is done, move to next

        # -------------------------------------------------
        # CHECK 2: LOW VARIANCE (Quasi-Constant)
        # -------------------------------------------------
        # We calculate the frequency of the most common value
        # dropna=True ensures we only look at actual data for this check
        value_counts = df[col].value_counts(normalize=True, dropna=True)
        
        if not value_counts.empty:
            most_common_val = value_counts.iloc[0] # Frequency of top value
            top_value_name = value_counts.index[0]
            
            if most_common_val > DOMINANCE_THRESHOLD:
                cols_to_drop.append(col)
                reasons[col] = f"Constant Value '{top_value_name}' in {most_common_val:.1%} of rows"

    # -------------------------------------------------
    # DELETE AND SAVE
    # -------------------------------------------------
    if cols_to_drop:
        print(f"  Found {len(cols_to_drop)} columns to remove:")
        for col in cols_to_drop:
            print(f"    - {col}: {reasons[col]}")
            
        df_clean = df.drop(columns=cols_to_drop)
        
        # Save to new file
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_cleaned.csv"
        
        df_clean.to_csv(output_filename, index=False)
        print(f"  Saved cleaned file: {output_filename}")
    else:
        print("  File is already clean (No columns met removal criteria).")
    
    print("-" * 40)

# ==========================================
# EXECUTION
# ==========================================

files = glob.glob(INPUT_PATTERN)

if not files:
    print(f"No files found matching '{INPUT_PATTERN}'")
else:
    for f in files:
        # Avoid processing files that are already cleaned
        if "_cleaned" in f:
            continue
        clean_dataset(f)