import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = 'ProductionConsumptionSettlement.csv'
OUTPUT_FOLDER = './energy_data_cleaned'
YEARS_TO_KEEP = [2024, 2025]
REGIONS = ['DK1', 'DK2']

def clean_and_save(df_subset, region, year):
    """
    Removes empty/constant columns BUT keeps 'Exchange' columns.
    """
    if df_subset.empty:
        print(f"⚠️  No data found for {region} {year}, skipping.")
        return

    print(f"\n🔍 INSPECTING: {region} - {year}")
    
    # 1. Identify Candidates for Removal
    # A. Empty Columns (All NaN)
    empty_candidates = [col for col in df_subset.columns if df_subset[col].isna().all()]
    
    # B. Constant Columns (Std Dev == 0)
    df_no_empty = df_subset.drop(columns=empty_candidates)
    numeric_cols = df_no_empty.select_dtypes(include=[np.number]).columns
    constant_candidates = []
    
    for col in numeric_cols:
        if df_no_empty[col].std() == 0:
            constant_candidates.append(col)

    # 2. Filter Candidates (The "Exchange" Safety Net)
    cols_to_drop = []
    
    print("   📢 TATTLETALE REPORT:")
    
    # Check Empty Candidates
    for col in empty_candidates:
        if "Exchange" in col:
            print(f"      🛡️  SAVED: {col:<30} (Was Empty, but is an Exchange column)")
        else:
            print(f"      ❌ DROPPED: {col:<30} (Reason: Entirely Empty)")
            cols_to_drop.append(col)
            
    # Check Constant Candidates
    for col in constant_candidates:
        unique_val = df_no_empty[col].iloc[0]
        if "Exchange" in col:
            print(f"      🛡️  SAVED: {col:<30} (Was Constant {unique_val}, but is an Exchange column)")
        else:
            print(f"      💤 DROPPED: {col:<30} (Reason: Constant Value {unique_val})")
            cols_to_drop.append(col)

    if not cols_to_drop:
        print("   ✅ No columns were removed.")

    # 3. Perform the actual removal
    df_cleaned = df_subset.drop(columns=cols_to_drop)

    # 4. Check for "Mostly Empty" Rows
    # Drop rows that are completely empty (ignoring metadata columns)
    initial_rows = len(df_cleaned)
    cols_to_check = df_cleaned.columns.difference(['HourUTC', 'HourDK', 'PriceArea', 'Year'])
    df_cleaned = df_cleaned.dropna(how='all', subset=cols_to_check)
    
    removed_rows = initial_rows - len(df_cleaned)
    if removed_rows > 0:
        print(f"      🗑️  Rows Removed: {removed_rows} (contained no data)")

    # 5. Save as STANDARD CSV (Comma separated, Dot decimal)
    filename = f"energy_{region}_{year}_cleaned.csv"
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    
    df_cleaned.to_csv(output_path, index=False, sep=',', decimal='.')
    print(f"   💾 Saved: {filename} (Columns: {len(df_subset.columns)} -> {len(df_cleaned.columns)})")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print(f"Loading {INPUT_FILE} with SEMICOLON separator...")
    
    try:
        # Load with semicolon separator and comma decimal (Danish format)
        df = pd.read_csv(INPUT_FILE, sep=';', decimal=',')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if len(df.columns) < 2:
        print("CRITICAL ERROR: Data loaded as 1 column. Check separators!")
        return
        
    # Parse Time
    df['HourDK'] = pd.to_datetime(df['HourDK'])
    df['Year'] = df['HourDK'].dt.year

    # Process splits
    for region in REGIONS:
        for year in YEARS_TO_KEEP:
            subset = df[
                (df['PriceArea'] == region) & 
                (df['Year'] == year)
            ].copy()
            
            clean_and_save(subset, region, year)

    print("\nDone! Files saved to:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()