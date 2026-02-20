import pandas as pd
import numpy as np
import os
import glob

# --- CONFIGURATION ---
INPUT_FOLDER = './Data'
# We want to find the files created by the previous script
SEARCH_PATTERN = os.path.join(INPUT_FOLDER, 'Production_DK*_202*.csv')

def clean_and_save(filepath):
    """
    Reads a pre-split file, removes empty/constant columns (keeps 'Exchange'),
    and saves it with a '_cleaned' suffix.
    """
    filename = os.path.basename(filepath)
    print(f"\n🔍 INSPECTING: {filename}")
    
    # Read the file (Note: The splitter saved it as standard CSV with ',' separator)
    df = pd.read_csv(filepath, sep=',')
    
    if df.empty:
        print("⚠️  File is empty, skipping.")
        return

    # 1. Identify Empty Columns (All NaN)
    empty_candidates = [col for col in df.columns if df[col].isna().all()]
    
    # 2. Identify Constant Columns (Std Dev == 0)
    df_no_empty = df.drop(columns=empty_candidates)
    numeric_cols = df_no_empty.select_dtypes(include=[np.number]).columns
    constant_candidates = []
    
    for col in numeric_cols:
        if df_no_empty[col].std() == 0:
            constant_candidates.append(col)

    # 3. Filter Candidates (The "Exchange" Safety Net)
    cols_to_drop = []
    print("   📢 TATTLETALE REPORT:")
    
    for col in empty_candidates:
        if "Exchange" in col:
            print(f"      🛡️  SAVED: {col:<30} (Was Empty, but is an Exchange column)")
        else:
            print(f"      ❌ DROPPED: {col:<30} (Reason: Entirely Empty)")
            cols_to_drop.append(col)
            
    for col in constant_candidates:
        unique_val = df_no_empty[col].iloc[0]
        if "Exchange" in col:
            print(f"      🛡️  SAVED: {col:<30} (Was Constant {unique_val}, but is an Exchange column)")
        else:
            print(f"      💤 DROPPED: {col:<30} (Reason: Constant Value {unique_val})")
            cols_to_drop.append(col)

    if not cols_to_drop:
        print("   ✅ No columns were removed.")

    # 4. Perform Removal
    df_cleaned = df.drop(columns=cols_to_drop)

    # 5. Check for "Mostly Empty" Rows
    initial_rows = len(df_cleaned)
    cols_to_check = df_cleaned.columns.difference(['HourUTC', 'HourDK', 'PriceArea', 'Year'])
    df_cleaned = df_cleaned.dropna(how='all', subset=cols_to_check)
    
    removed_rows = initial_rows - len(df_cleaned)
    if removed_rows > 0:
        print(f"      🗑️  Rows Removed: {removed_rows} (contained no data)")

    # 6. Save File
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}_cleaned.csv"
    output_path = os.path.join(INPUT_FOLDER, output_filename)
    
    df_cleaned.to_csv(output_path, index=False, sep=',', decimal='.')
    print(f"   💾 Saved: {output_filename} (Columns: {len(df.columns)} -> {len(df_cleaned.columns)})")

def main():
    # Find all the split files
    files_to_clean = glob.glob(SEARCH_PATTERN)
    
    # Filter out files that already have '_cleaned' in the name so we don't process them twice
    files_to_clean = [f for f in files_to_clean if '_cleaned' not in f]
    
    if not files_to_clean:
        print(f"No files found matching {SEARCH_PATTERN}. Run the Splitter first!")
        return

    print(f"Found {len(files_to_clean)} files to clean.")
    
    for file in files_to_clean:
        clean_and_save(file)

    print("\nDone! Cleaned files are in:", INPUT_FOLDER)

if __name__ == "__main__":
    main()