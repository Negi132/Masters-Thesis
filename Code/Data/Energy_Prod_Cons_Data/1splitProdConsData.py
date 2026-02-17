import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "ProductionConsumptionSettlement.csv"

# Define the time ranges as requested
# 2024: 2024-01-01 00:00 to 2025-01-01 00:00
START_2024 = pd.Timestamp("2024-01-01 00:00:00")
END_2024   = pd.Timestamp("2025-01-01 00:00:00")

# 2025: 2025-01-01 00:00 to 2026-01-01 00:00
START_2025 = pd.Timestamp("2025-01-01 00:00:00")
END_2025   = pd.Timestamp("2026-01-01 00:00:00")

def process_energy_data():
    print(f"Reading {INPUT_FILE}...")
    
    # Read CSV (EnergiNet uses ';' separator and ',' for decimals)
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', decimal=',')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Convert Timestamp to datetime objects
    print("Parsing timestamps...")
    df['HourUTC'] = pd.to_datetime(df['HourUTC'])

    # Sort by time initially (helper step)
    df.sort_values(by='HourUTC', inplace=True)

    # ==========================================
    # PROCESS BY REGION
    # ==========================================
    
    for area in ['DK1', 'DK2']:
        print(f"\nProcessing Region: {area}")
        
        # Filter by Area
        df_area = df[df['PriceArea'] == area].copy()
        
        # ----------------------------------
        # SPLIT INTO 2024
        # ----------------------------------
        # Condition: Start <= t <= End (Inclusive of end as requested)
        mask_2024 = (df_area['HourUTC'] >= START_2024) & (df_area['HourUTC'] <= END_2024)
        df_2024 = df_area[mask_2024].copy()
        
        if not df_2024.empty:
            filename_2024 = f"Production_{area}_2024.csv"
            # Ensure strictly sorted by time
            df_2024.sort_values(by='HourUTC', ascending=True, inplace=True)
            df_2024.to_csv(filename_2024, index=False, sep=',')
            print(f"  -> Saved {filename_2024} ({len(df_2024)} rows)")
        else:
            print(f"  -> Warning: No data found for {area} in 2024 range.")

        # ----------------------------------
        # SPLIT INTO 2025
        # ----------------------------------
        mask_2025 = (df_area['HourUTC'] >= START_2025) & (df_area['HourUTC'] <= END_2025)
        df_2025 = df_area[mask_2025].copy()
        
        if not df_2025.empty:
            filename_2025 = f"Production_{area}_2025.csv"
            df_2025.sort_values(by='HourUTC', ascending=True, inplace=True)
            df_2025.to_csv(filename_2025, index=False, sep=',')
            print(f"  -> Saved {filename_2025} ({len(df_2025)} rows)")
        else:
            print(f"  -> Warning: No data found for {area} in 2025 range.")

if __name__ == "__main__":
    process_energy_data()