import json
import os
from datetime import datetime, timezone

# --- CONFIGURATION ---
DATA_DIRECTORY = './data'
FILE_EXTENSIONS = ('.txt', '.json', '.jsonl')

def normalize_to_utc(timestamp_str):
    """
    Parses a timestamp string, converts it to UTC, and returns
    the ISO 8601 string with a +00:00 offset.
    """
    if not timestamp_str:
        return timestamp_str
    
    try:
        # Parse the ISO format string (handles existing offsets automatically)
        dt = datetime.fromisoformat(timestamp_str)
        
        # Convert to UTC
        dt_utc = dt.astimezone(timezone.utc)
        
        # Return formatted string (isoformat adds +00:00)
        return dt_utc.isoformat()
    except ValueError:
        # If parsing fails, return original string
        return timestamp_str

def process_file(filepath):
    """
    Reads a file line-by-line, converts timestamps to UTC, 
    and writes to a temporary file before replacing the original.
    """
    temp_filepath = filepath + '.tmp'
    modified_count = 0
    line_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as infile, \
         open(temp_filepath, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            try:
                data = json.loads(line)
                
                # Check for the 'properties' dictionary
                if 'properties' in data:
                    props = data['properties']
                    
                    # Fields to standardise
                    time_fields = ['from', 'to', 'created', 'calculatedAt', 'observed']
                    
                    for field in time_fields:
                        if field in props:
                            original_time = props[field]
                            new_time = normalize_to_utc(original_time)
                            
                            # Only update if the string actually changes
                            if original_time != new_time:
                                props[field] = new_time
                                modified_count += 1
                                
                # Write the data back
                outfile.write(json.dumps(data) + '\n')
                
            except json.JSONDecodeError:
                # If a line isn't valid JSON, write it back as-is
                outfile.write(line + '\n')

    # Replace original file with the fixed version
    os.replace(temp_filepath, filepath)
    print(f"Processed {os.path.basename(filepath)}: {line_count} lines, {modified_count} timestamps updated.")

def main():
    if not os.path.exists(DATA_DIRECTORY):
        print(f"Directory not found: {DATA_DIRECTORY}")
        return

    print(f"Standardizing timestamps to UTC in: {DATA_DIRECTORY}")
    
    files_found = False
    for filename in os.listdir(DATA_DIRECTORY):
        if filename.endswith(FILE_EXTENSIONS):
            files_found = True
            filepath = os.path.join(DATA_DIRECTORY, filename)
            process_file(filepath)
            
    if not files_found:
        print("No matching files found.")
    else:
        print("Done! All timestamps are now consistent UTC.")

if __name__ == "__main__":
    main()