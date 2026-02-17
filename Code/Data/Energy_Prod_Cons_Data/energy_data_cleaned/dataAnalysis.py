import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

# PATH ADJUSTMENT:
# Point this to the folder containing your moved CSV/JSON files.
# "." means current folder, "./weather_data" means a subfolder.
DATA_FOLDER = "." 

# The name of the report file that will be generated
OUTPUT_REPORT_FILE = "production_consumption_data_analysis_report.txt"

# Which files should we look for?
# We use "**" to look inside all subfolders recursively.
FILE_PATTERNS = [
    "**/*_cleaned.csv",       # Your final cleaned weather data
    "**/*_production.csv",    # Your production models
    "**/*.json"               # The external consultant files
]

# ==========================================
# LOGGING HELPER
# ==========================================

def log(message, file_handle):
    """Prints to console AND writes to the report file."""
    print(message)
    file_handle.write(message + "\n")

# ==========================================
# ANALYSIS FUNCTIONS
# ==========================================

def analyze_file(filepath, report_file):
    log("=" * 85, report_file)
    log(f"ANALYZING: {os.path.basename(filepath)}", report_file)
    log(f"Full Path: {filepath}", report_file)
    log("-" * 85, report_file)

    # 1. LOAD DATA
    # ------------------------------------------------
    try:
        if filepath.endswith(".json"):
            # Load JSON (Handle the MIDAS/Consultant format)
            with open(filepath, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            
            # Normalize column names for consistency
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'Timestamp_UTC'}, inplace=True)
                
        else:
            # Load CSV (Our created files)
            df = pd.read_csv(filepath)
    except Exception as e:
        log(f"ERROR: Could not read file. Reason: {e}", report_file)
        return

    # 2. BASIC DIMENSIONS
    # ------------------------------------------------
    num_rows, num_cols = df.shape
    log(f"Dimensions:      {num_rows:,} rows x {num_cols} columns", report_file)
    
    mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    log(f"Memory Usage:    {mem_usage:.2f} MB", report_file)

    # 3. TIME COVERAGE
    # ------------------------------------------------
    time_col = None
    for col in ['Timestamp_UTC', 'time', 'datetime', 'Timestamp']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            start_time = df[time_col].min()
            end_time = df[time_col].max()
            duration = end_time - start_time
            
            log(f"Time Range:      {start_time}  to  {end_time}", report_file)
            log(f"Duration:        {duration}", report_file)
            
            # Check for missing hours (gaps)
            expected_hours = int(duration.total_seconds() / 3600) + 1
            missing_hours = expected_hours - num_rows
            
            if missing_hours > 0:
                log(f"WARNING:         Potential gap of {missing_hours} hours detected!", report_file)
            elif missing_hours == 0:
                log("Integrity:       No missing hours detected (Continuous).", report_file)
            else:
                log(f"Note:            {abs(missing_hours)} duplicate/extra timestamps found.", report_file)
        except:
            log("Time Parsing:    Could not parse time column.", report_file)
    else:
        log("Time Column:     Not found.", report_file)

    # 4. MISSING VALUES ANALYSIS
    # ------------------------------------------------
    total_cells = num_rows * num_cols
    if total_cells == 0:
        log("Error: Empty dataset.", report_file)
        return

    total_missing = df.isna().sum().sum()
    total_missing_pct = (total_missing / total_cells) * 100
    
    log("-" * 40, report_file)
    log(f"Overall Dataset Missing Data: {total_missing_pct:.2f}%", report_file)
    log("-" * 40, report_file)
    
    log("Feature Statistics (Missing % | Mean | Min | Max):", report_file)
    log(f"{'Column Name':<30} | {'Miss %':<8} | {'Mean':<10} | {'Min':<10} | {'Max':<10}", report_file)
    log("-" * 85, report_file)

    # Select only numeric columns for stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in df.columns:
        # Missing %
        missing = df[col].isna().mean() * 100
        
        # Stats (only if numeric)
        if col in numeric_cols:
            mean_val = df[col].mean()
            min_val = df[col].min()
            max_val = df[col].max()
            
            # Format nicely
            mean_str = f"{mean_val:.2f}"
            min_str = f"{min_val:.2f}"
            max_str = f"{max_val:.2f}"
        else:
            mean_str = "-"
            min_str = "-"
            max_str = "-"
            
        log(f"{col:<30} | {missing:>7.2f}% | {mean_str:>10} | {min_str:>10} | {max_str:>10}", report_file)

    log("\n", report_file)

# ==========================================
# EXECUTION LOOP
# ==========================================

def run_analysis():
    print(f"Scanning for files in: {os.path.abspath(DATA_FOLDER)}")
    
    found_files = []
    for pattern in FILE_PATTERNS:
        # recursive=True allows "**" to work
        full_pattern = os.path.join(DATA_FOLDER, pattern)
        found_files.extend(glob.glob(full_pattern, recursive=True))
    
    # Remove duplicates and sort
    found_files = sorted(list(set(found_files)))

    if not found_files:
        print("No files found! Check your DATA_FOLDER path.")
        return

    print(f"Found {len(found_files)} files. Writing report to {OUTPUT_REPORT_FILE}...")
    
    # Open the report file once and append all analyses to it
    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as report:
        log("*************************************************************************************", report)
        log("                     PRODUCTION/CONSUMPTION DATA ANALYSIS REPORT                              ", report)
        log("*************************************************************************************", report)
        log(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report)
        log(f"Source Folder: {os.path.abspath(DATA_FOLDER)}", report)
        log("\n", report)

        for filepath in found_files:
            analyze_file(filepath, report)
            
    print(f"Done! Report saved as: {OUTPUT_REPORT_FILE}")

if __name__ == "__main__":
    run_analysis()