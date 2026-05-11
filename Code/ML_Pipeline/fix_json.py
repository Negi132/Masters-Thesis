import json
import os

def fix_json_keys(filename):
    if not os.path.exists(filename):
        print(f"Could not find {filename}")
        return
    
    with open(filename, 'r') as f:
        data = json.load(f)
        
    new_data = {}
    for key, val in data.items():
        # Find where the horizon string starts (e.g., "_24h") and cut the string off there
        new_key = key
        for tag in ["_0h", "_24h", "_48h", "_72h"]:
            if tag in new_key:
                new_key = new_key.split(tag)[0]
                break
        
        new_data[new_key] = val
        
    # Overwrite the file with the clean keys
    with open(filename, 'w') as f:
        json.dump(new_data, f, indent=4)
    print(f"Successfully cleaned keys in {filename}!")
        
if __name__ == "__main__":
    fix_json_keys("pruned_features_Price.json")
    fix_json_keys("pruned_features_Delta.json")
    print("All JSON files are ready for the Pruning Engine!")