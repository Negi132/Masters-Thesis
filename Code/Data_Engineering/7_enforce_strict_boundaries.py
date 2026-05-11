import os
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")
ALIGNED_DIR = BASE_PATH / "Aligned_Yearly"
DMI_ALIGNED_DIR = BASE_PATH / "DMI" / "AlignedZones"

def enforce_common_boundaries():
    print("Starting Strict Boundary Enforcement...")
    print("-" * 60)
    
    dmi_files = sorted(DMI_ALIGNED_DIR.glob("*_aligned.csv"))
    if not dmi_files:
        print("[ERROR] No DMI files found.")
        return

    trimmed_count = 0

    for dmi_file in dmi_files:
        parts = dmi_file.name.split('_')
        year, region = parts[0], parts[1]
        identifier = f"{year}_{region}"
        
        price_file = ALIGNED_DIR / f"Prices_{identifier}.csv"
        prod_file = ALIGNED_DIR / f"ProdCons_{identifier}.csv"

        if not price_file.exists() or not prod_file.exists():
            continue

        # Load datasets
        df_dmi = pd.read_csv(dmi_file)
        df_price = pd.read_csv(price_file)
        df_prod = pd.read_csv(prod_file)

        # Convert to Datetime for accurate comparison
        df_dmi['HourUTC_dt'] = pd.to_datetime(df_dmi['HourUTC'])
        df_price['HourUTC_dt'] = pd.to_datetime(df_price['HourUTC'])
        df_prod['HourUTC_dt'] = pd.to_datetime(df_prod['HourUTC'])

        # Find absolute limits
        start_dmi, end_dmi = df_dmi['HourUTC_dt'].min(), df_dmi['HourUTC_dt'].max()
        start_price, end_price = df_price['HourUTC_dt'].min(), df_price['HourUTC_dt'].max()
        start_prod, end_prod = df_prod['HourUTC_dt'].min(), df_prod['HourUTC_dt'].max()

        # The Intersection logic: Latest Start, Earliest End
        common_start = max(start_dmi, start_price, start_prod)
        common_end = min(end_dmi, end_price, end_prod)

        # Check if the files are already perfectly aligned
        is_aligned = (
            start_dmi == common_start and end_dmi == common_end and
            start_price == common_start and end_price == common_end and
            start_prod == common_start and end_prod == common_end
        )

        if not is_aligned:
            print(f"Trimming {identifier} overhangs...")
            print(f"  -> Snapping to intersection: {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
            
            # Apply strict bounding box
            df_dmi = df_dmi[(df_dmi['HourUTC_dt'] >= common_start) & (df_dmi['HourUTC_dt'] <= common_end)]
            df_price = df_price[(df_price['HourUTC_dt'] >= common_start) & (df_price['HourUTC_dt'] <= common_end)]
            df_prod = df_prod[(df_prod['HourUTC_dt'] >= common_start) & (df_prod['HourUTC_dt'] <= common_end)]

            # Drop the temporary datetime column
            df_dmi = df_dmi.drop(columns=['HourUTC_dt'])
            df_price = df_price.drop(columns=['HourUTC_dt'])
            df_prod = df_prod.drop(columns=['HourUTC_dt'])

            # Overwrite the files with the perfectly aligned data
            df_dmi.to_csv(dmi_file, index=False)
            df_price.to_csv(price_file, index=False)
            df_prod.to_csv(prod_file, index=False)
            
            trimmed_count += 1

    print("-" * 60)
    if trimmed_count > 0:
        print(f"Successfully trimmed {trimmed_count} datasets to their common intersections.")
    else:
        print("All datasets were already perfectly aligned.")
        
    print("Run the Audit Script (7) again to verify 100% synchronization!")

if __name__ == "__main__":
    enforce_common_boundaries()