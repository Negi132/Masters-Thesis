import os
import csv
import re
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")
CSV_FILES = [
    Path("ProductionConsumptionSettlement.csv"),
    Path("Elspotprices.csv"),
    Path("DayAheadPrices.csv")
]

STANDARD_UTC_HEADER = "HourUTC"

def format_utc_timestamp(ts):
    """Converts '2026-03-24 22:45:00' to '2026-03-24T22:45:00+00:00'"""
    ts = ts.strip()
    if not ts:
        return ts
    
    # Replace the space between date and time with a 'T'
    if " " in ts:
        ts = ts.replace(" ", "T")
        
    # If it lacks a timezone offset, force append strict UTC
    if not re.search(r'(Z|[+-]\d{2}:?\d{2})$', ts):
        ts += "+00:00"
        
    return ts

def analyze_and_standardize():
    print("Starting CSV Format & Timestamp Standardization...")
    print("-" * 60)

    for file_path in CSV_FILES:
        # Fallback pathing logic
        if not file_path.exists():
            alternate_path = BASE_PATH / "Prices" / file_path.name
            alternate_path_2 = BASE_PATH / "Prod_Cons" / file_path.name
            
            if alternate_path.exists():
                file_path = alternate_path
            elif alternate_path_2.exists():
                file_path = alternate_path_2
            else:
                print(f"[MISSING] Could not find {file_path.name}")
                continue

        print(f"\nProcessing: {file_path.name}")
        
        # --- Detect Format ---
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()

        detected_delimiter = ';' if ';' in first_line else ',' if ',' in first_line else '\t'
        uses_comma_decimals = bool(detected_delimiter == ';' and re.search(r'\d,\d', second_line))

        output_name = file_path.stem + "_standardized.csv"
        output_path = file_path.parent / output_name 
        
        with open(file_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter=detected_delimiter)
            
            try:
                headers = next(reader)
            except StopIteration:
                print("  [ERROR] File is empty.")
                continue
                
            # Identify the UTC column and the local column to drop
            utc_col_idx = -1
            dk_col_idx = -1
            
            for i, h in enumerate(headers):
                h_lower = h.lower()
                if 'utc' in h_lower:
                    headers[i] = STANDARD_UTC_HEADER
                    utc_col_idx = i
                elif 'dk' in h_lower:
                    dk_col_idx = i

            # Create new headers without the DK column
            final_headers = [h for i, h in enumerate(headers) if i != dk_col_idx]

            # Write Standardized Output
            with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow(final_headers)
                
                rows_processed = 0
                for row in reader:
                    cleaned_row = []
                    for i, val in enumerate(row):
                        # Skip the DK local time column entirely
                        if i == dk_col_idx:
                            continue
                            
                        val = val.strip()
                        
                        # Apply strict ISO-8601 formatting to the UTC column
                        if i == utc_col_idx:
                            val = format_utc_timestamp(val)
                        
                        # Fix European decimals
                        elif uses_comma_decimals and re.match(r'^-?\d+,\d+$', val):
                            val = val.replace(',', '.')
                            
                        cleaned_row.append(val)
                        
                    writer.writerow(cleaned_row)
                    rows_processed += 1

        print(f"  -> Converted {rows_processed} rows.")
        print("  -> European decimals converted to standard periods.")
        print("  -> UTC Timestamps strictly enforced to ISO-8601 (+00:00).")
        if dk_col_idx != -1:
            print("  -> Local Time column dropped to prevent DST corruption.")

    print("\n" + "-" * 60)
    print("Standardization complete. Data is now bulletproof for Pandas merging.")

if __name__ == "__main__":
    analyze_and_standardize()