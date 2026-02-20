import pandas as pd
import numpy as np
import glob
import os
import json
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

# We define exactly what files belong to which domain, and where to save the report
DOMAINS = {
    "Weather": {
        "patterns": [
            "./DMI/Data/*_timeseries_cleaned.csv",  # Your previously cleaned CSVs
            "./DMI/Data/*.json"             # The newly added Midas JSON files
        ],
        "output": "data_analysis_report_weather.txt"
    },
    "Energy": {
        "patterns": [
            "./Energy/Data/*_cleaned.csv"
        ],
        "output": "data_analysis_report_energy.txt"
    },
    "Prices": {
        "patterns": [
            "./Prices/Data/*_cleaned.csv"
        ],
        "output": "data_analysis_report_price.txt"
    }
}

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
            
            # Normalize column names so our time-check works on everything
            if 'datetime' in df.columns:
                df.rename(columns={'datetime': 'HourUTC'}, inplace=True)
                
        else:
            # Load CSV (Our standard output)
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

    # 3. TIME COVERAGE & GAP DETECTION
    # ------------------------------------------------
    time_col = None
    # Look for known time columns across our 3 domains
    for col in ['HourUTC', 'Timestamp_UTC', 'time', 'datetime', 'Timestamp']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col:
        try:
            # utc=True standardizes everything and prevents offset bugs
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            start_time = df[time_col].min()
            end_time = df[time_col].max()
            duration = end_time - start_time
            
            log(f"Time Range:      {start_time.strftime('%Y-%m-%d %H:%M:%S')}  to  {end_time.strftime('%Y-%m-%d %H:%M:%S')}", report_file)
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
        except Exception as e:
            log(f"Time Parsing:    Could not parse time column '{time_col}'. ({e})", report_file)
    else:
        log("Time Column:     Not found.", report_file)

    # 4. MISSING VALUES & FEATURE STATISTICS
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
    
    log("Feature Statistics (Numeric Summary):", report_file)
    
    # Header format tailored for 85 width
    log(f"{'Column Name':<32} | {'Miss %':<8} | {'Mean':<12} | {'Min':<12} | {'Max':<12}", report_file)
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
            
        log(f"{col:<32} | {missing:>7.2f}% | {mean_str:>12} | {min_str:>12} | {max_str:>12}", report_file)

    log("\n", report_file)

# ==========================================
# EXECUTION LOOP
# ==========================================

def run_analysis():
    print("Starting Comprehensive Data Analysis...\n")
    
    for domain, config in DOMAINS.items():
        found_files = []
        for pattern in config["patterns"]:
            # Search for files
            found_files.extend(glob.glob(pattern, recursive=True))
        
        # Remove duplicates and sort
        found_files = sorted(list(set(found_files)))
        
        if not found_files:
            print(f"⚠️ No files found for {domain} matching patterns.")
            continue
            
        output_file = config["output"]
        print(f"-> Found {len(found_files)} files for {domain}. Writing report to {output_file}...")
        
        # Open report file for this specific domain
        with open(output_file, "w", encoding="utf-8") as report:
            log("*" * 85, report)
            log(f"{domain.upper()} DATA ANALYSIS REPORT".center(85), report)
            log("*" * 85, report)
            log(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", report)
            log("\n", report)

            for filepath in found_files:
                analyze_file(filepath, report)
                
        print(f"   ✅ {domain} analysis complete!\n")

if __name__ == "__main__":
    run_analysis()