import pandas as pd
import os
from pathlib import Path

def main():
    print("--------------------------------------------------")
    print("INITIALIZING HORIZON MATRIX GENERATION")
    print("--------------------------------------------------")
    
    # Dynamically find the path based on the script location
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = current_dir / "Data" / "ML_Ready_Data"
    
    # Define the regions and horizons to process
    regions = ['DK1', 'DK2']
    horizons = [24, 48, 72, 96, 120, 144, 168]
    
    for region in regions:
        print(f"\n================ Processing Region: {region} ================")
        base_file = base_dir / f"Master_Matrix_{region}_Horizon0h.csv"
        
        if not base_file.exists():
            print(f"[WARNING] Could not find the base matrix at: {base_file}")
            print(f"Skipping {region}...")
            continue
            
        print(f"Loading base matrix: {base_file.name}")
        df_base = pd.read_csv(base_file)
        
        # Sort by time to ensure shifting operations are mathematically sound
        if 'HourUTC' in df_base.columns:
            df_base['HourUTC'] = pd.to_datetime(df_base['HourUTC'])
            df_base = df_base.sort_values('HourUTC').reset_index(drop=True)
            
        # Determine the base price column to use for shifting
        price_col = 'TARGET_Price_0h' if 'TARGET_Price_0h' in df_base.columns else 'SpotPriceEUR'
        
        for h in horizons:
            print(f"  Generating Master Matrix for Horizon: {h}h...")
            df_h = df_base.copy()
            
            target_price_col = f'TARGET_Price_{h}h'
            target_delta_col = f'TARGET_Delta_{h}h'
            
            # Shift the target price backward by 'h' rows to simulate the future
            df_h[target_price_col] = df_h[price_col].shift(-h)
            
            # Delta = Future Price - Current Price
            df_h[target_delta_col] = df_h[target_price_col] - df_h[price_col]
            
            # Shifting introduces NaN values at the very end of the dataset. Drop them.
            df_h = df_h.dropna(subset=[target_price_col, target_delta_col]).reset_index(drop=True)
            
            # Save the specific horizon matrix
            output_file = base_dir / f"Master_Matrix_{region}_Horizon{h}h.csv"
            df_h.to_csv(output_file, index=False)
            print(f"    -> Saved {output_file.name} | Total Rows: {len(df_h)}")
            
    print("\n--------------------------------------------------")
    print("ALL HORIZON MATRICES GENERATED SUCCESSFULLY")
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()