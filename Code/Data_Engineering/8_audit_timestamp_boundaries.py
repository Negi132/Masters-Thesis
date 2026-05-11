import os
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")
ALIGNED_DIR = BASE_PATH / "Aligned_Yearly"
DMI_ALIGNED_DIR = BASE_PATH / "DMI" / "AlignedZones"

def audit_boundaries():
    print("Starting Master Timestamp Boundary Audit...")
    print("=" * 90)
    print(f"{'YEAR/ZONE':<12} | {'DMI BOUNDS':<25} | {'PRICE BOUNDS':<25} | {'PROD/CONS BOUNDS':<25}")
    print("-" * 90)

    # Get a list of all aligned DMI files to act as our baseline
    dmi_files = sorted(DMI_ALIGNED_DIR.glob("*_aligned.csv"))
    
    if not dmi_files:
        print("[ERROR] No DMI files found in AlignedZones.")
        return

    mismatch_found = False

    for dmi_file in dmi_files:
        # Extract metadata (e.g., '2024', 'DK1')
        parts = dmi_file.name.split('_')
        year, region = parts[0], parts[1]
        identifier = f"{year}_{region}"
        
        # Define paths for the other two domains
        price_file = ALIGNED_DIR / f"Prices_{identifier}.csv"
        prod_file = ALIGNED_DIR / f"ProdCons_{identifier}.csv"

        try:
            # Load the first and last row of each file to check boundaries efficiently
            df_dmi = pd.read_csv(dmi_file, usecols=['HourUTC'])
            dmi_start, dmi_end = df_dmi['HourUTC'].min(), df_dmi['HourUTC'].max()

            if price_file.exists():
                df_price = pd.read_csv(price_file, usecols=['HourUTC'])
                price_start, price_end = df_price['HourUTC'].min(), df_price['HourUTC'].max()
            else:
                price_start, price_end = "MISSING", "MISSING"

            if prod_file.exists():
                df_prod = pd.read_csv(prod_file, usecols=['HourUTC'])
                prod_start, prod_end = df_prod['HourUTC'].min(), df_prod['HourUTC'].max()
            else:
                prod_start, prod_end = "MISSING", "MISSING"

            # Check for perfect alignment
            is_aligned = (dmi_start == price_start == prod_start) and (dmi_end == price_end == prod_end)
            
            # Format strings for clean printing (just taking the MMDD for brevity if needed, but full ISO is safer)
            dmi_str = f"{dmi_start[:10]} to {dmi_end[:10]}" if dmi_start != "MISSING" else "MISSING"
            price_str = f"{price_start[:10]} to {price_end[:10]}" if price_start != "MISSING" else "MISSING"
            prod_str = f"{prod_start[:10]} to {prod_end[:10]}" if prod_start != "MISSING" else "MISSING"

            if is_aligned:
                print(f"{identifier:<12} | {dmi_str:<25} | {price_str:<25} | {prod_str:<25} [OK]")
            else:
                print(f"{identifier:<12} | {dmi_str:<25} | {price_str:<25} | {prod_str:<25} [MISMATCH]")
                mismatch_found = True

        except Exception as e:
            print(f"{identifier:<12} | ERROR: {e}")

    print("=" * 90)
    if mismatch_found:
        print("[WARNING] Critical boundary mismatches detected! Files do not perfectly align.")
    else:
        print("[SUCCESS] All files across all domains have perfectly synchronized start and end boundaries.")

if __name__ == "__main__":
    audit_boundaries()