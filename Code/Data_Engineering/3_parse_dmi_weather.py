import os
import json
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("Data")
DMI_RAW_DIR = BASE_PATH / "DMI" / "CombinedYearlyRaw"
DMI_OUTPUT_DIR = BASE_PATH / "DMI" / "ProcessedZones"

# DENMARK MAINLAND BOUNDING BOX (Excludes Greenland, Faroe Islands, & Bornholm)
MIN_LON = 8.0
MAX_LON = 13.0  
MIN_LAT = 54.5
MAX_LAT = 58.0

LONGITUDE_CUTOFF = 11.0

# We use sets for O(1) fast lookups
DK1_STATION_IDS = {
    "06030", "06041", "06049", "06051", "06052", "06056", "06058", "06060", 
    "06065", "06068", "06070", "06072", "06073", "06074", "06079", "06080", 
    "06081", "06082", "06088", "06093", "06096", "06102", "06104", "06110", 
    "06116", "06118", "06119", "06120", "06123", "06124", "06126", "06132"
}

# NOTE: Bornholm stations (e.g., 06190, 06193, 06197) should be removed from this 
# set if you want to strictly exclude them from DK2 averages!
DK2_STATION_IDS = {
    "06135", "06136", "06138", "06141", "06147", "06149", "06151", "06154", 
    "06156", "06159", "06160", "06165", "06168", "06169", "06170", "06174", 
    "06180", "06181", "06183", "06184", "06186", "06187", "06188"
#    , "06190", "06193", "06197"
}

# ==========================================
# PROCESSING FUNCTIONS
# ==========================================

def get_zone(station_id, coordinates):
    """
    Determines if a record belongs to DK1 or DK2.
    1. Check explicit Station ID list.
    2. Fallback to geographically shielded bounding box.
    """
    if station_id in DK1_STATION_IDS:
        return "DK1"
    if station_id in DK2_STATION_IDS:
        return "DK2"
    
    # Fallback: Use Coordinates [lon, lat]
    if coordinates and len(coordinates) >= 2:
        try:
            lon = float(coordinates[0])
            lat = float(coordinates[1])
            
            # The Geographic Shield
            if (MIN_LON <= lon <= MAX_LON) and (MIN_LAT <= lat <= MAX_LAT):
                if lon < LONGITUDE_CUTOFF:
                    return "DK1"
                else:
                    return "DK2"
        except (ValueError, TypeError):
            pass
            
    return "UNKNOWN"

def process_dmi_year(file_path):
    year = file_path.stem
    print(f"\nProcessing Year: {year}...")
    
    records_dk1 = []
    records_dk2 = []
    unknown_stations = set()
    fallback_stations = set()
    skipped_resolutions = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                props = obj.get('properties', {})
                
                if props.get('timeResolution') != 'hour':
                    skipped_resolutions += 1
                    continue
                
                station_id = props.get('stationId')
                param_id = props.get('parameterId')
                raw_time = props.get('from')
                value = props.get('value')
                
                if value is None or not raw_time or not station_id:
                    continue
                    
                coordinates = obj.get('geometry', {}).get('coordinates', [])
                zone = get_zone(station_id, coordinates)
                
                record = {
                    'RawTime': raw_time,
                    'Parameter': param_id,
                    'Value': float(value),
                    'StationId': station_id
                }
                
                if zone == "DK1":
                    records_dk1.append(record)
                    if station_id not in DK1_STATION_IDS: fallback_stations.add(station_id)
                elif zone == "DK2":
                    records_dk2.append(record)
                    if station_id not in DK2_STATION_IDS: fallback_stations.add(station_id)
                else:
                    unknown_stations.add(station_id)
                    
            except (json.JSONDecodeError, ValueError):
                continue

    if fallback_stations:
        print(f"  [INFO] Rescued {len(fallback_stations)} mainland stations using Bounding Box.")
    if unknown_stations:
        print(f"  [WARN] Ignored {len(unknown_stations)} stations (Greenland, Faroe, Bornholm, or unknown).")

    for zone_name, records in [("DK1", records_dk1), ("DK2", records_dk2)]:
        if not records: continue
        df = pd.DataFrame(records)
        df['HourUTC'] = pd.to_datetime(df['RawTime'], utc=True).dt.floor('h')
        
        grouped = df.groupby(['HourUTC', 'Parameter']).agg(
            Value_Avg=('Value', 'mean'), Station_Count=('StationId', 'nunique')
        ).reset_index()
        
        pivot_vals = grouped.pivot(index='HourUTC', columns='Parameter', values='Value_Avg')
        qc_counts = grouped.groupby('HourUTC')['Station_Count'].agg(
            Stations_Reporting_Min='min', Stations_Reporting_Max='max'
        )
        
        final_df = pivot_vals.join(qc_counts).sort_index(ascending=True)
        final_df.index = final_df.index.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        final_df.index.name = 'HourUTC'
        
        output_filename = DMI_OUTPUT_DIR / f"{year}_{zone_name}_hourly.csv"
        final_df.to_csv(output_filename)
        print(f"     Saved {zone_name}: {output_filename.name} ({len(final_df)} hours)")

if __name__ == "__main__":
    print("Starting Geographically Shielded DMI Parser...")
    DMI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(DMI_RAW_DIR.glob("*.txt")):
        process_dmi_year(file_path)