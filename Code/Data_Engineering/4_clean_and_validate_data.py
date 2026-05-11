import os
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")

TARGET_FILES = [
    BASE_PATH / "Prod_Cons" / "ProductionConsumptionSettlement_standardized.csv",
    BASE_PATH / "Prices" / "Elspotprices_standardized.csv",
    BASE_PATH / "Prices" / "DayAheadPrices_standardized.csv"
]
DMI_DIR = BASE_PATH / "DMI" / "ProcessedZones"

MISSING_THRESHOLD = 0.90 
DOMINANCE_THRESHOLD = 0.95

ALWAYS_KEEP = [
    'HourUTC', 
    'PriceArea',
    'SpotPriceEUR',
    'DayAheadPriceEUR'
]

REPORT_FILE = "data_validation_report.txt"

# ==========================================
# PROCESSING FUNCTION
# ==========================================
def validate_and_clean_file(file_path, report_lines):
    if not file_path.exists():
        return
        
    report_lines.append(f"\nAnalyzing: {file_path.name}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        report_lines.append(f"  [ERROR] Reading file: {e}")
        return

    # --- NEW: STEP 0: Filter PriceArea ---
    if 'PriceArea' in df.columns:
        initial_rows = len(df)
        df = df[df['PriceArea'].isin(['DK1', 'DK2'])]
        filtered_rows = len(df)
        report_lines.append(f"  Filtered PriceArea: Kept {filtered_rows} DK1/DK2 rows (Dropped {initial_rows - filtered_rows} foreign rows).")
    
    rows = len(df)
    if rows == 0:
        report_lines.append("  [ERROR] File is empty or no DK1/DK2 rows remain. Skipping file.")
        return

    cols_to_drop = []
    reasons = {}

    # --- STEP 1: Identify Bad Columns ---
    for col in df.columns:
        # Protect explicitly kept columns AND any column containing "exchange"
        if col in ALWAYS_KEEP or 'exchange' in col.lower():
            continue
            
        missing_ratio = df[col].isna().sum() / rows
        if missing_ratio > MISSING_THRESHOLD:
            cols_to_drop.append(col)
            reasons[col] = f"Missing {missing_ratio:.1%} of data."
            continue
            
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            value_counts = valid_data.value_counts(normalize=True)
            most_common_ratio = value_counts.iloc[0]
            top_value = value_counts.index[0]
            
            if most_common_ratio > DOMINANCE_THRESHOLD:
                cols_to_drop.append(col)
                reasons[col] = f"Constant value '{top_value}' in {most_common_ratio:.1%} of rows."

    # --- STEP 2: Drop Bad Columns ---
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        report_lines.append(f"  Dropped {len(cols_to_drop)} columns:")
        for col in cols_to_drop:
            report_lines.append(f"    - {col}: {reasons[col]}")
    else:
        report_lines.append("  No columns required dropping.")

    # --- NEW: STEP 3A: Zero-Fill Exchange Columns ---
    exchange_cols = [col for col in df.columns if 'exchange' in col.lower()]
    if exchange_cols:
        missing_exchange = df[exchange_cols].isna().sum().sum()
        if missing_exchange > 0:
            df[exchange_cols] = df[exchange_cols].fillna(0)
            report_lines.append(f"  Zero-filled {missing_exchange} missing values across {len(exchange_cols)} Exchange columns.")

    # --- STEP 3B: Time-Aware Interpolation & Flagging (For non-exchange numeric columns) ---
    all_numeric_cols = df.select_dtypes(include=['number']).columns
    # Isolate columns that need strict mathematical interpolation
    numeric_cols = [col for col in all_numeric_cols if 'exchange' not in col.lower()]
    
    missing_before = df[numeric_cols].isna().sum().sum() if numeric_cols else 0
    
    if missing_before > 0:
        # A. Create Imputation Flags (Shadow Variables)
        flagged_count = 0
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[f"{col}_imputed"] = df[col].isna().astype(int)
                flagged_count += 1
                
        # B. Convert to Datetime and set as Index
        if 'HourUTC' in df.columns:
            df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True)
            df.set_index('HourUTC', inplace=True)
            time_indexed = True
        else:
            time_indexed = False

        # C. 1st Pass: Strict Time-Based Interpolation (max 12 hour gap)
        if time_indexed:
            df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit=12, limit_direction='both')
        else:
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=12, limit_direction='both')
            
        # D. 2nd Pass: Forward/Backward fill for edge cases (Capped at 12 hours)
        df[numeric_cols] = df[numeric_cols].ffill(limit=12).bfill(limit=12)
        
        # E. 3rd Pass: Column Mean Failsafe
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                col_mean = df[col].mean()
                df[col] = df[col].fillna(col_mean)

        # F. Reset index and restore strict string formatting
        if time_indexed:
            df.reset_index(inplace=True)
            df['HourUTC'] = df['HourUTC'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')

        report_lines.append(f"  Interpolated/Filled {missing_before} numerical values.")
        report_lines.append(f"  Created binary '_imputed' flags for {flagged_count} columns.")

    # --- STEP 4: Save Cleaned File ---
    output_name = file_path.stem + "_validated.csv"
    output_name = output_name.replace("_standardized_validated", "_validated")
    output_path = file_path.parent / output_name
    
    df.to_csv(output_path, index=False)
    report_lines.append(f"  Saved validated data to: {output_name}")
    report_lines.append("-" * 40)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Starting Advanced Data Validation Protocol...")
    report_lines = []
    report_lines.append("==================================================")
    report_lines.append("          DATA VALIDATION & CLEANING REPORT")
    report_lines.append("==================================================")
    
    for file_path in TARGET_FILES:
        validate_and_clean_file(file_path, report_lines)
        
    if DMI_DIR.exists():
        dmi_files = sorted(DMI_DIR.glob("*_hourly.csv"))
        for dmi_file in dmi_files:
            validate_and_clean_file(dmi_file, report_lines)
    else:
        report_lines.append(f"\n[WARN] DMI directory not found: {DMI_DIR}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"Validation complete! Please check '{REPORT_FILE}' for details.")

if __name__ == "__main__":
    main()