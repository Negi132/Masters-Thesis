import os
import glob

def merge_txt_files(input_folder, output_filename, file_pattern="*.txt"):
    """
    Merges all files matching the pattern in the input_folder into one output_filename.
    Uses streaming to handle large files without crashing RAM.
    """
    # Get all matching files
    files_to_merge = sorted(glob.glob(os.path.join(input_folder, file_pattern)))
    
    if not files_to_merge:
        print(f"No files found in {input_folder} matching {file_pattern}")
        return

    print(f"Found {len(files_to_merge)} files. Starting merge into {output_filename}...")

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for i, fname in enumerate(files_to_merge):
            print(f"[{i+1}/{len(files_to_merge)}] Appending: {os.path.basename(fname)}")
            
            with open(fname, 'r', encoding='utf-8') as infile:
                # We stream the file line by line to keep memory usage near zero
                for line in infile:
                    outfile.write(line)
                
                # Ensure each file ends with a newline so the next file starts correctly
                # (Optional: Only if you suspect files might be missing the final newline)
                # outfile.write('\n')

    print(f"Finished! Combined file saved as: {output_filename}")

# --- EXECUTION ---
# Change these paths to match where you saved your DMI downloads
# Example: If your files are in folders 'data_2024' and 'data_2025'

# Merge 2024
merge_txt_files(
    input_folder='./data/2024station/', 
    output_filename='./data/dmi_full_2024_raw.txt'
)

# Merge 2025
merge_txt_files(
    input_folder='./data/2025station', 
    output_filename='./data/dmi_full_2025_raw.txt'
)