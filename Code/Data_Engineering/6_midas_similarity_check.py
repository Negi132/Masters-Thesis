import pandas as pd
import os
import json

# ==========================================
# CONFIGURATION
# ==========================================
# Point these to the specific year/region you want to test
CSV_PATH = "./Data/DMI/ProcessedZones/2024_DK1_hourly_validated.csv"
JSON_PATH = "./Data/weather_dk1_2024Midas.json"

# Explicit pairs to check: (Your_CSV_Column, MIDAS_JSON_Column)
PAIRS_TO_CHECK = [
    ('temp_dry', 'avg_temp_dry'),
    ('wind_speed', 'avg_wind_speed'),
    ('mean_pressure', 'avg_pressure'),
    ('mean_cloud_cover', 'avg_cloud_cover'),
    ('bright_sunshine', 'avg_sun_last1h_glob'),
    ('radia_glob', 'avg_radia_glob_past1h')
]

# MIDAS targets to hunt for the best correlation
MIDAS_TARGETS = [
    'avg_temp_dry', 'avg_wind_speed', 'avg_wind_dir', 
    'avg_humidity', 'avg_cloud_cover', 'avg_sun_last1h_glob', 
    'avg_radia_glob_past1h'
]

def load_data():
    if not os.path.exists(CSV_PATH) or not os.path.exists(JSON_PATH):
        print("Error: Could not find CSV or JSON file. Check paths.")
        return None, None
        
    df_csv = pd.read_csv(CSV_PATH)
    df_csv['HourUTC'] = pd.to_datetime(df_csv['HourUTC'], utc=True)
    
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df_midas = pd.DataFrame(data)
    
    if 'datetime' in df_midas.columns:
        df_midas['HourUTC'] = pd.to_datetime(df_midas['datetime'], utc=True)
        
    return df_csv, df_midas

def run_diagnostics():
    df_csv, df_midas = load_data()
    if df_csv is None: return

    print("Merging data on HourUTC...")
    df_merged = pd.merge(df_csv, df_midas, on='HourUTC', how='inner')
    
    if df_merged.empty:
        print("Error: No matching timestamps found between datasets.")
        return

    print("\n" + "="*80)
    print("PART 1: DIRECT PAIR COMPARISON")
    print("="*80)
    print(f"{'CSV COLUMN':<20} | {'MIDAS COLUMN':<22} | {'CORR':<6} | {'MAE':<8} | {'SCALE FACTOR'}")
    print("-" * 80)

    for col_csv, col_midas in PAIRS_TO_CHECK:
        if col_csv not in df_merged.columns or col_midas not in df_merged.columns:
            print(f"{col_csv:<20} | {col_midas:<22} | {'MISSING':<6} | {'-':<8} | -")
            continue
            
        valid_data = df_merged[[col_csv, col_midas]].dropna()
        if valid_data.empty: continue

        c_csv = valid_data[col_csv]
        c_midas = valid_data[col_midas]

        corr = c_csv.corr(c_midas)
        mae = (c_csv - c_midas).abs().mean()
        
        scale = c_csv.mean() / c_midas.mean() if c_midas.mean() != 0 else 0
        
        print(f"{col_csv:<20} | {col_midas:<22} | {corr:>6.3f} | {mae:>8.2f} | {scale:.2f}x")

    print("\n" + "="*80)
    print("PART 2: FIND BEST MATCHING FEATURES")
    print("="*80)
    print(f"{'MIDAS TARGET':<22} | {'YOUR BEST MATCH':<25} | {'CORR':<6} | {'MEAN DIFF'}")
    print("-" * 80)

    # Get all numeric columns from CSV to test against
    csv_numeric_cols = df_csv.select_dtypes(include=['number']).columns

    for target in MIDAS_TARGETS:
        if target not in df_merged.columns:
            continue
            
        best_col = None
        best_corr = -1
        mean_diff = 0
        
        for candidate in csv_numeric_cols:
            if candidate not in df_merged.columns: continue
            
            corr = df_merged[target].corr(df_merged[candidate])
            if pd.notna(corr) and abs(corr) > best_corr:
                best_corr = abs(corr)
                best_col = candidate
                mean_diff = df_merged[candidate].mean() - df_merged[target].mean()

        if best_col:
            print(f"{target:<22} | {best_col:<25} | {best_corr:>6.3f} | {mean_diff:>8.2f}")

if __name__ == "__main__":
    run_diagnostics()