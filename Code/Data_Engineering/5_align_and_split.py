import os
import re
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")

PROD_CONS_FILE = BASE_PATH / "Prod_Cons" / "ProductionConsumptionSettlement_validated.csv"
SPOT_PRICE_FILE = BASE_PATH / "Prices" / "Elspotprices_validated.csv"
DAY_AHEAD_FILE = BASE_PATH / "Prices" / "DayAheadPrices_validated.csv"

# Input and Output for DMI Alignment
DMI_DIR = BASE_PATH / "DMI" / "ProcessedZones"
DMI_ALIGNED_DIR = BASE_PATH / "DMI" / "AlignedZones"

OUTPUT_DIR = BASE_PATH / "Aligned_Yearly"
REPORT_FILE = "alignment_and_split_report.txt"

def enforce_dmi_consistency(report_lines):
    print("Step 1: Enforcing DMI Column Consistency (Non-Destructive)...")
    report_lines.append("--- STEP 1: DMI COLUMN CONSISTENCY ---")
    
    dmi_files = list(DMI_DIR.glob("*_validated.csv"))
    
    if not dmi_files:
        msg = "  [ERROR] No validated DMI files found."
        print(msg)
        report_lines.append(msg)
        return []

    DMI_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Find the intersection AND union of all columns
    common_columns = None
    all_columns = set()
    
    for f in dmi_files:
        cols = set(pd.read_csv(f, nrows=0).columns)
        all_columns.update(cols)
        if common_columns is None:
            common_columns = cols
        else:
            common_columns = common_columns.intersection(cols)
            
    # The Validation AI "Tattle-Tale" Logic
    dropped = all_columns - common_columns
    if dropped:
        msg = f"  [WARN] {len(dropped)} columns dropped due to cross-year inconsistency: {sorted(dropped)}"
        print(msg)
        report_lines.append(msg)
    else:
        msg = "  [INFO] All DMI files have perfectly consistent columns. No features dropped."
        print(msg)
        report_lines.append(msg)
            
    # Keep HourUTC first, then sort the rest alphabetically for perfect consistency
    final_cols = ['HourUTC'] + sorted([c for c in common_columns if c != 'HourUTC'])
    msg_cols = f"  -> Preserving {len(final_cols)} common columns across all DMI files."
    print(msg_cols)
    report_lines.append(msg_cols)

    # 2. Rewrite DMI files to a NEW directory
    aligned_files = []
    for f in dmi_files:
        df = pd.read_csv(f)
        df = df[final_cols]
        df = df.sort_values(by='HourUTC', ascending=True)
        
        output_path = DMI_ALIGNED_DIR / f.name.replace("_validated", "_aligned")
        df.to_csv(output_path, index=False)
        aligned_files.append(output_path)
        
    return aligned_files

def combine_prices(report_lines):
    print("\nStep 2: Combining, Aggregating, and Sorting Price Files...")
    report_lines.append("\n--- STEP 2: PRICE COMBINATION & AGGREGATION ---")
    
    df_spot = pd.read_csv(SPOT_PRICE_FILE)
    df_da = pd.read_csv(DAY_AHEAD_FILE)
    
    # Safely convert to datetime objects
    df_spot['HourUTC'] = pd.to_datetime(df_spot['HourUTC'], utc=True)
    df_da['HourUTC'] = pd.to_datetime(df_da['HourUTC'], utc=True)
    
    # --- THE 15-MINUTE RESOLUTION FIX ---
    # 1. Force all timestamps to round down to the nearest hour
    df_spot['HourUTC'] = df_spot['HourUTC'].dt.floor('h')
    df_da['HourUTC'] = df_da['HourUTC'].dt.floor('h')
    
    # 2. Group by the new hourly timestamp and average the prices
    # ADDED numeric_only=True to prevent crashes on leftover string columns
    df_spot = df_spot.groupby(['HourUTC', 'PriceArea'], as_index=False).mean(numeric_only=True)
    df_da = df_da.groupby(['HourUTC', 'PriceArea'], as_index=False).mean(numeric_only=True)
    
    # Merge on Time and Area
    df_combined = pd.merge(df_spot, df_da, on=['HourUTC', 'PriceArea'], how='outer')
    
    # Reorder columns to be logical and consistent
    price_cols = ['HourUTC', 'PriceArea', 'SpotPriceEUR', 'DayAheadPriceEUR']
    extra_cols = sorted([c for c in df_combined.columns if c not in price_cols])
    df_combined = df_combined[price_cols + extra_cols]
    
    df_combined = df_combined.sort_values(by=['PriceArea', 'HourUTC'], ascending=True)
    
    msg = f"  -> Aggregated and combined prices into master frame with {len(df_combined)} rows."
    print(msg)
    report_lines.append(msg)
    
    return df_combined

def split_datasets(aligned_dmi_files, df_prices, report_lines):
    print("\nStep 3: Splitting Prices and Prod/Cons by Year and Region...")
    report_lines.append("\n--- STEP 3: YEARLY & REGIONAL SPLITS ---")
    
    df_prod = pd.read_csv(PROD_CONS_FILE)
    df_prod['HourUTC'] = pd.to_datetime(df_prod['HourUTC'], utc=True)
    df_prod = df_prod.sort_values(by=['PriceArea', 'HourUTC'], ascending=True)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for dmi_file in aligned_dmi_files:
        # Robust Regex Parsing for Year and Region
        match = re.search(r'(\d{4})_(DK[12])', dmi_file.name)
        if not match:
            msg = f"  [WARN] Skipping {dmi_file.name} - Does not match expected YYYY_DKX format."
            print(msg)
            report_lines.append(msg)
            continue
            
        year = match.group(1)
        region = match.group(2)
        
        # Read DMI file and convert to datetime for safe temporal math
        df_dmi = pd.read_csv(dmi_file)
        df_dmi['HourUTC'] = pd.to_datetime(df_dmi['HourUTC'], utc=True)
        
        start_date = df_dmi['HourUTC'].min()
        end_date = df_dmi['HourUTC'].max()
        
        # Safe Datetime Filtering
        mask_prices = (df_prices['PriceArea'] == region) & (df_prices['HourUTC'] >= start_date) & (df_prices['HourUTC'] <= end_date)
        prices_subset = df_prices[mask_prices].copy()
        
        mask_prod = (df_prod['PriceArea'] == region) & (df_prod['HourUTC'] >= start_date) & (df_prod['HourUTC'] <= end_date)
        prod_subset = df_prod[mask_prod].copy()
        
        # Restore strict ISO-8601 string formatting before saving
        prices_subset['HourUTC'] = prices_subset['HourUTC'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        prod_subset['HourUTC'] = prod_subset['HourUTC'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        
        # Save Outputs
        prices_subset.to_csv(OUTPUT_DIR / f"Prices_{year}_{region}.csv", index=False)
        prod_subset.to_csv(OUTPUT_DIR / f"ProdCons_{year}_{region}.csv", index=False)
        
        msg = f"  -> {year} {region}: Generated Prices ({len(prices_subset)} rows), ProdCons ({len(prod_subset)} rows)"
        print(msg)
        report_lines.append(msg)

def main():
    print("Starting Bulletproof Alignment and Split Protocol...")
    print("-" * 60)
    
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("      ALIGNMENT & SPLIT DIAGNOSTIC REPORT")
    report_lines.append("==================================================")
    
    aligned_dmi_files = enforce_dmi_consistency(report_lines)
    if aligned_dmi_files:
        df_prices = combine_prices(report_lines)
        split_datasets(aligned_dmi_files, df_prices, report_lines)
        
    print("-" * 60)
    print(f"Alignment and Splitting Complete! Check '{REPORT_FILE}' for the dropped column log.")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    main()