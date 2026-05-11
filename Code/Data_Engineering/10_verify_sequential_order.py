import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")
ML_DATA_DIR = BASE_PATH / "ML_Ready_Data"

def verify_timeline(file_path):
    print(f"\nAnalyzing: {file_path.name}")
    
    # We only need the timestamp column for this check (super fast)
    df = pd.read_csv(file_path, usecols=['HourUTC'])
    df['HourUTC'] = pd.to_datetime(df['HourUTC'])
    
    # 1. Monotonic Check (Strictly forward in time?)
    is_monotonic = df['HourUTC'].is_monotonic_increasing
    print(f"  -> Strictly Sequential: {'[PASS]' if is_monotonic else '[FAIL] Backward jumps detected!'}")
    
    # 2. Duplicate Check
    duplicates = df['HourUTC'].duplicated().sum()
    print(f"  -> Duplicate Timestamps: {duplicates} {'[PASS]' if duplicates == 0 else '[FAIL]'}")
    
    # 3. Continuity Check (Exactly 1-hour steps?)
    time_diffs = df['HourUTC'].diff().dropna()
    gaps = (time_diffs != pd.Timedelta(hours=1)).sum()
    
    if gaps == 0:
        print("  -> Continuous 1-Hour Steps: [PASS] (No missing hours!)")
    else:
        print(f"  -> Continuity Warning: Found {gaps} jumps that are NOT exactly 1 hour.")
        # If there are gaps, let's print the biggest one so you know how bad it is
        max_gap = time_diffs.max()
        print(f"     Largest single gap: {max_gap}")

def main():
    print("==================================================")
    print("      TIMELINE INTEGRITY VERIFICATION")
    print("==================================================")
    
    matrix_files = list(ML_DATA_DIR.glob("Master_Matrix_*.csv"))
    
    if not matrix_files:
        print("[ERROR] No Master Matrix files found.")
        return
        
    for file in matrix_files:
        verify_timeline(file)
        
    print("\n==================================================")
    print("Verification complete.")

if __name__ == "__main__":
    main()