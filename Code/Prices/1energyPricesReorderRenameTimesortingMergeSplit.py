import pandas as pd
import os

# --- CONFIGURATION ---
FILE_BEFORE = './Data/ElspotpricesBefore.csv'  # The "Before" file (SpotPrice naming)
FILE_AFTER = './Data/DayAheadPricesAfter.csv'  # The "After" file (DayAhead naming)
OUTPUT_FOLDER = './Data/'

def process_prices():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("Loading price files...")
    
    # 1. Load the "Before" file (Elspotprices)
    # This file has columns: HourUTC, HourDK, PriceArea, SpotPriceDKK, SpotPriceEUR
    try:
        df_before = pd.read_csv(FILE_BEFORE, sep=';', decimal=',')
    except Exception as e:
        print(f"Error reading {FILE_BEFORE}: {e}")
        return

    # 2. Load the "After" file (DayAheadPrices)
    # This file has columns: TimeUTC, TimeDK, PriceArea, DayAheadPriceEUR, DayAheadPriceDKK
    try:
        df_after = pd.read_csv(FILE_AFTER, sep=';', decimal=',')
    except Exception as e:
        print(f"Error reading {FILE_AFTER}: {e}")
        return

    print(f" - Before File: {len(df_before)} rows")
    print(f" - After File:  {len(df_after)} rows")

    # 3. Standardize Columns in the "After" dataframe
    # We rename 'DayAhead...' to 'SpotPrice...' and 'Time...' to 'Hour...'
    rename_map = {
        'TimeUTC': 'HourUTC',
        'TimeDK': 'HourDK',
        'DayAheadPriceEUR': 'SpotPriceEUR',
        'DayAheadPriceDKK': 'SpotPriceDKK'
    }
    df_after.rename(columns=rename_map, inplace=True)

    # 4. Enforce Column Order
    # We use the column order from the 'Before' file as the standard
    target_order = ['HourUTC', 'HourDK', 'PriceArea', 'SpotPriceDKK', 'SpotPriceEUR']
    
    # Check if all columns exist
    for col in target_order:
        if col not in df_before.columns:
            print(f"Error: Column {col} missing in {FILE_BEFORE}")
            return
        if col not in df_after.columns:
            print(f"Error: Column {col} missing in {FILE_AFTER} (after renaming)")
            return

    # Reorder both just to be safe
    df_before = df_before[target_order]
    df_after = df_after[target_order]

    # 5. Merge and Sort
    print("Merging and sorting...")
    full_df = pd.concat([df_before, df_after], ignore_index=True)
    
    # Parse dates for sorting
    full_df['HourUTC'] = pd.to_datetime(full_df['HourUTC'])
    full_df.sort_values(by='HourUTC', ascending=True, inplace=True)

    # Optional: Remove duplicates if the files overlap
    initial_len = len(full_df)
    full_df.drop_duplicates(subset=['HourUTC', 'PriceArea'], keep='last', inplace=True)
    if len(full_df) < initial_len:
        print(f" - Removed {initial_len - len(full_df)} duplicate rows.")

    # 6. Split by Region (DK1 / DK2) and Save
    regions = ['DK1', 'DK2']
    
    for region in regions:
        region_df = full_df[full_df['PriceArea'] == region].copy()
        
        if region_df.empty:
            print(f"Warning: No data found for {region}")
            continue
            
        filename = f"elspotprices_{region}.csv"
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        # Save as standard CSV (comma separated, dot decimal)
        region_df.to_csv(output_path, index=False, sep=',', decimal='.')
        print(f"✅ Saved: {filename} ({len(region_df)} rows)")

    print("\nDone! Processed price files are in:", OUTPUT_FOLDER)

if __name__ == "__main__":
    process_prices()