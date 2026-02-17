import json
import os
import glob
from collections import defaultdict
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

# Format of the timestamp in your DMI files
# Example: "2024-01-01T00:00:00+00:00"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S" 

# Which parameters do you want to keep?
# (Empty list = Keep ALL parameters found)
KEEP_PARAMS = [
#    "wind_speed",       # Wind Speed
#    "wind_dir",         # Wind Direction
#    "temp_dry",         # Temperature
#    "radia_glob",       # Solar Radiation
#    "mean_cloud_cover"  # Cloud Cover
]

# ==========================================
# PROCESSING FUNCTION
# ==========================================

def get_hourly_timestamp(iso_string):
    """
    Truncates a timestamp to the specific hour.
    Input: "2024-01-01T14:35:00+00:00" -> Output: "2024-01-01T14:00:00"
    """
    try:
        # Split at '+' to ignore timezone for simplicity, or keep it if needed.
        # DMI data is usually UTC (+00:00).
        clean_str = iso_string.split('+')[0].split('.')[0] # Remove TZ and microseconds
        dt = datetime.strptime(clean_str, DATE_FORMAT)
        
        # Set minutes/seconds to 0 to group by hour
        dt_hour = dt.replace(minute=0, second=0)
        return dt_hour.isoformat()
    except Exception:
        return None

def process_station_averages(input_filename):
    print(f"Reading file: {input_filename} ...")
    print("  (This requires RAM to hold hourly stats. Please wait.)")

    # Data Structure:
    # keys: (station_id, timestamp_hour, parameter_id)
    # values: [sum_of_values, count_of_values]
    aggregated_data = defaultdict(lambda: [0.0, 0])
    
    line_count = 0
    
    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            if line_count % 500000 == 0:
                print(f"  Processed {line_count} lines...")

            try:
                data = json.loads(line)
                props = data.get("properties", {})
                
                # Extract Key Information
                station_id = props.get("stationId")
                param_id = props.get("parameterId")
                value = props.get("value")
                time_from = props.get("from")
                
                # Filter unwanted parameters (if list is set)
                if KEEP_PARAMS and param_id not in KEEP_PARAMS:
                    continue

                if station_id and param_id and value is not None and time_from:
                    # Get the hour bucket
                    hour_key = get_hourly_timestamp(time_from)
                    
                    if hour_key:
                        # Add to aggregator
                        entry = aggregated_data[(station_id, hour_key, param_id)]
                        entry[0] += float(value) # Sum
                        entry[1] += 1            # Count
                        
            except (json.JSONDecodeError, ValueError):
                continue

    # ==========================================
    # WRITE OUTPUT
    # ==========================================
    
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_hourly_avg.csv"
    
    print(f"  Writing results to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8') as out:
        # Write CSV Header
        out.write("station_id,timestamp_utc,parameter,value_avg,count\n")
        
        # Sort keys for cleaner output (Station -> Time -> Param)
        sorted_keys = sorted(aggregated_data.keys())
        
        for key in sorted_keys:
            station, time, param = key
            total_sum, count = aggregated_data[key]
            
            avg_value = total_sum / count
            
            out.write(f"{station},{time},{param},{avg_value:.4f},{count}\n")

    print(f"Finished {input_filename}!")
    print("-" * 40)

# ==========================================
# EXECUTION
# ==========================================

# Find the files from the previous step (e.g. "dmi_2024_DK1.txt")
# You can adjust the pattern "*.txt" or "*.jsonl" as needed.
files_to_process = sorted(glob.glob("*_DK*.txt"))

if not files_to_process:
    print("No DK1/DK2 files found! Please place this script in the data folder.")
else:
    for fname in files_to_process:
        # Avoid processing files we just created
        if "_hourly_avg" in fname:
            continue
        process_station_averages(fname)