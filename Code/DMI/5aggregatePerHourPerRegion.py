import csv
import glob
import os
from collections import defaultdict

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FOLDER = './Data'

# The columns to exclude from the averaging (metadata)
# We group by timestamp, so we don't average that.
IGNORE_COLS = {'station_id', 'timestamp_utc', 'count'}

# ==========================================
# PROCESSING FUNCTION
# ==========================================

def calculate_regional_averages(input_filename):
    print(f"Processing regional stats for: {input_filename} ...")
    
    # Data Structure:
    # timestamps[time_str][parameter_name] = [sum_of_values, count_of_stations]
    timestamps = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    
    # Set to keep track of all unique parameters found in this file (e.g. wind_speed, temp_dry)
    all_parameters = set()

    # 1. READ DATA
    with open(input_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            ts = row['timestamp_utc']
            param = row['parameter']
            
            try:
                val = float(row['value_avg'])
            except ValueError:
                continue
                
            # Add to the regional accumulator
            stats = timestamps[ts][param]
            stats[0] += val # Sum
            stats[1] += 1   # Count (number of stations contributing to this hour)
            
            all_parameters.add(param)

    # 2. CALCULATE AVERAGES & WRITE OUTPUT
    
    # Sort parameters so columns are always in the same order (e.g. alphabetical)
    param_list = sorted(list(all_parameters))
    
    # Output filename: e.g. "2024_DK1_regional_timeseries.csv"
    base_name = os.path.splitext(input_filename)[0].replace('_hourly_avg', '')
    output_filename = f"{base_name}_regional_timeseries.csv"
    
    print(f"  Writing regional timeseries to {output_filename}...")
    
    with open(output_filename, 'w', encoding='utf-8', newline='') as f_out:
        # Create Header: Timestamp + each Parameter
        header = ['Timestamp_UTC'] + param_list + ['Stations_Reporting_Min', 'Stations_Reporting_Max']
        writer = csv.writer(f_out)
        writer.writerow(header)
        
        # Sort by time
        sorted_times = sorted(timestamps.keys())
        
        for ts in sorted_times:
            row_data = [ts]
            
            # Metadata to track data quality (how many stations are alive this hour)
            min_stations = 9999
            max_stations = 0
            
            for param in param_list:
                if param in timestamps[ts]:
                    total_sum, count = timestamps[ts][param]
                    avg_val = total_sum / count
                    row_data.append(f"{avg_val:.4f}")
                    
                    # Update station counts for quality check
                    if count < min_stations: min_stations = count
                    if count > max_stations: max_stations = count
                else:
                    # Missing data for this specific parameter at this hour
                    row_data.append("") 
            
            # Append the station counts (useful for debugging if data drops out)
            if min_stations == 9999: min_stations = 0
            row_data.append(min_stations)
            row_data.append(max_stations)
            
            writer.writerow(row_data)

    print(f"Done! Created {output_filename}")
    print("-" * 40)

# ==========================================
# EXECUTION
# ==========================================

# Find the files from the PREVIOUS step (must end in _hourly_avg.csv)
search_pattern = os.path.join(INPUT_FOLDER, "*_hourly_avg.csv")
files_to_process = glob.glob(search_pattern)

if not files_to_process:
    print("No station-average files found matching '*_hourly_avg.csv'.")
    print("Please run the previous 'average per station' script first.")
else:
    for fname in files_to_process:
        calculate_regional_averages(fname)