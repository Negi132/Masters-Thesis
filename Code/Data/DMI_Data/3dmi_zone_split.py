import json
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================

# DK1 (West Denmark): Jylland (Jutland) + Fyn (Funen)
# DK2 (East Denmark): Sjælland (Zealand) + Bornholm + Islands
# We use a set for O(1) fast lookups
DK1_STATION_IDS = {
    "06030", "06041", "06049", "06051", "06052", "06056", "06058", "06060", 
    "06065", "06068", "06070", "06072", "06073", "06074", "06079", "06080", 
    "06081", "06082", "06088", "06093", "06096", "06102", "06104", "06110", 
    "06116", "06118", "06119", "06120", "06123", "06124", "06126", "06132"
}

DK2_STATION_IDS = {
    "06135", "06136", "06138", "06141", "06147", "06149", "06151", "06154", 
    "06156", "06159", "06160", "06165", "06168", "06169", "06170", "06174", 
    "06180", "06181", "06183", "06184", "06186", "06187", "06188", "06190", 
    "06193", "06197"
}

# Fallback: Longitude cutoff (approx 11.0° East separates DK1 and DK2)
LONGITUDE_CUTOFF = 11.0

# ==========================================
# PROCESSING FUNCTION
# ==========================================

def get_zone(station_id, coordinates):
    """
    Determines if a record belongs to DK1 or DK2.
    1. Check explicit Station ID list.
    2. Fallback to longitude check if ID is unknown.
    """
    # Check ID first (fast & accurate)
    if station_id in DK1_STATION_IDS:
        return "DK1"
    if station_id in DK2_STATION_IDS:
        return "DK2"
    
    # Fallback: Use Longitude from coordinates [lon, lat]
    if coordinates and len(coordinates) >= 1:
        try:
            lon = float(coordinates[0])
            if lon < LONGITUDE_CUTOFF:
                return "DK1"
            else:
                return "DK2"
        except (ValueError, TypeError):
            pass
            
    return "UNKNOWN"

def split_file_by_zone(input_filename):
    """
    Reads a JSONL file line-by-line and writes to two separate output files.
    """
    base_name = os.path.splitext(input_filename)[0]
    output_dk1 = f"{base_name}_DK1.txt"  # Keeping .txt extension as per your source
    output_dk2 = f"{base_name}_DK2.txt"

    print(f"Processing: {input_filename}")
    print(f" -> Creating: {output_dk1} & {output_dk2}")

    count_dk1 = 0
    count_dk2 = 0
    count_unknown = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_dk1, 'w', encoding='utf-8') as f_dk1, \
             open(output_dk2, 'w', encoding='utf-8') as f_dk2:
            
            for line_num, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue

                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Extract identifying info
                    props = data.get("properties", {})
                    geometry = data.get("geometry", {})
                    
                    station_id = props.get("stationId")
                    coords = geometry.get("coordinates") # Expecting [lon, lat]

                    # Determine Zone
                    zone = get_zone(station_id, coords)

                    # Write to appropriate file
                    if zone == "DK1":
                        f_dk1.write(line + '\n')
                        count_dk1 += 1
                    elif zone == "DK2":
                        f_dk2.write(line + '\n')
                        count_dk2 += 1
                    else:
                        # Optional: Log unknown stations if needed
                        count_unknown += 1

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON at line {line_num+1}")
                    continue

        print(f"Done! Stats for {input_filename}:")
        print(f"  DK1 Records: {count_dk1}")
        print(f"  DK2 Records: {count_dk2}")
        if count_unknown > 0:
            print(f"  Unknown/Skipped: {count_unknown}")
        print("-" * 40)

    except Exception as e:
        print(f"Error processing {input_filename}: {e}")

# ==========================================
# EXECUTION
# ==========================================

# Pattern to find your aggregated files (e.g., "dmi_full_2024_raw.txt")
# You can change this pattern to match whatever you named them in the previous script.
files_to_process = sorted(glob.glob("dmi_full_*_raw.txt"))

if not files_to_process:
    print("No aggregated files found matching 'dmi_full_*_raw.txt'.")
    print("Please ensure this script is in the same folder as your aggregated files.")
else:
    for fname in files_to_process:
        split_file_by_zone(fname)