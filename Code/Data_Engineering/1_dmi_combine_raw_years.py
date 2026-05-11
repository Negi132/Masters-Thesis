import os
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# Assuming this script is run from the 'Refracturing' directory.
# Adjust the BASE_PATH if you place the script elsewhere.
BASE_PATH = Path("Data/DMI")
RAW_DIR = BASE_PATH / "YearlyRaw"
COMBINED_DIR = BASE_PATH / "CombinedYearlyRaw"

def combine_dmi_yearly_data():
    print("Starting DMI raw data consolidation...")
    
    # 1. Ensure the output directory exists
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RAW_DIR.exists():
        print(f"Error: Could not find the raw data directory at {RAW_DIR}")
        print("Please check your paths and run again.")
        return

    # 2. Identify all yearly subfolders (e.g., 2015, 2016...)
    year_folders = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    year_folders.sort()

    if not year_folders:
        print(f"No yearly folders found in {RAW_DIR}")
        return

    # 3. Iterate through each year and combine its daily files
    for year_folder in year_folders:
        year = year_folder.name
        output_filepath = COMBINED_DIR / f"{year}.txt"
        
        # Get all daily text files in this folder, sorted chronologically
        daily_files = sorted(year_folder.glob("*.txt"))
        
        if not daily_files:
            print(f"Skipping {year}: No text files found.")
            continue
            
        print(f"Processing year {year}... ({len(daily_files)} files found)")
        
        # 4. Open the output file in write mode
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            for daily_file in daily_files:
                # Open each daily file in read mode
                with open(daily_file, 'r', encoding='utf-8') as infile:
                    # Write line-by-line to handle large files efficiently without memory crashes
                    for line in infile:
                        # Ensure we don't end up with concatenated lines if a file lacks a trailing newline
                        outfile.write(line)
                        if not line.endswith('\n'):
                            outfile.write('\n')
                            
        print(f"Successfully created: {output_filepath}")

    print("\nConsolidation complete. All yearly files are ready in Data/DMI/CombinedYearlyRaw/")

if __name__ == "__main__":
    combine_dmi_yearly_data()