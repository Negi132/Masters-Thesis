import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# This matches the output folder from the previous script
INPUT_FOLDER = './Data/'
# The specific files we expect to find
TARGET_FILES = ['elspotprices_DK1.csv', 'elspotprices_DK2.csv']

def clean_and_save(filename):
    filepath = os.path.join(INPUT_FOLDER, filename)
    
    if not os.path.exists(filepath):
        print(f"⚠️  Skipping {filename}: File not found in {INPUT_FOLDER}")
        return

    print(f"\n🔍 INSPECTING: {filename}")
    
    # Load the file (assuming standard CSV format from previous step)
    try:
        df = pd.read_csv(filepath, sep=',', decimal='.')
    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return

    # 1. Identify Candidates for Removal
    # A. Empty Columns (All NaN)
    empty_candidates = [col for col in df.columns if df[col].isna().all()]
    
    # B. Constant Columns (Std Dev == 0)
    # We only check numeric columns for constants (preserving string IDs like 'PriceArea')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_candidates = []
    
    for col in numeric_cols:
        if df[col].std() == 0:
            constant_candidates.append(col)

    # 2. Tattletale Report
    cols_to_drop = []
    
    if empty_candidates or constant_candidates:
        print("   📢 TATTLETALE REPORT (Columns Removed):")
        
        for col in empty_candidates:
            print(f"      ❌ {col:<25} -> REASON: Entirely Empty (All NaN)")
            cols_to_drop.append(col)
            
        for col in constant_candidates:
            unique_val = df[col].iloc[0]
            print(f"      💤 {col:<25} -> REASON: Constant Value (Always {unique_val})")
            cols_to_drop.append(col)
    else:
        print("   ✅ No boring or empty columns found.")

    # 3. Perform Removal
    if cols_to_drop:
        df_cleaned = df.drop(columns=cols_to_drop)
    else:
        df_cleaned = df.copy()

    # 4. Save with _cleaned suffix
    # e.g. elspotprices_DK1.csv -> elspotprices_DK1_cleaned.csv
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_cleaned.csv"
    output_path = os.path.join(INPUT_FOLDER, output_filename)
    
    df_cleaned.to_csv(output_path, index=False, sep=',', decimal='.')
    print(f"   💾 Saved: {output_filename} (Columns: {len(df.columns)} -> {len(df_cleaned.columns)})")

def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Directory {INPUT_FOLDER} does not exist.")
        return

    print(f"Cleaning price files in: {INPUT_FOLDER}...")
    
    files_processed = 0
    for filename in TARGET_FILES:
        clean_and_save(filename)
        files_processed += 1

    if files_processed == 0:
        print("No matching files found to clean.")
    else:
        print("\nDone!")

if __name__ == "__main__":
    main()