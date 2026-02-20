import os
import subprocess

# Define the folders and their master scripts
PIPELINES = [
    ("DMI", "0runPreprocessing.py"),
    ("Energy", "0runPreprocessing.py"),
    ("Prices", "0runPreprocessing.py")
]

def run_pipeline(folder, script_name):
    print(f"\n{'='*50}")
    print(f"🚀 STARTING PIPELINE: {folder.upper()}")
    print(f"{'='*50}")
    
    script_path = os.path.join(folder, script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Could not find {script_path}")
        return
        
    try:
        # cwd=folder makes Python pretend the terminal is inside that folder!
        subprocess.run(["python", script_name], cwd=folder, check=True)
        print(f"\n✅ Successfully completed {folder} pipeline.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name} in {folder}. (Exit code: {e.returncode})")

def main():
    print("Initializing Master Preprocessing Run...")
    
    for folder, script in PIPELINES:
        run_pipeline(folder, script)

    print("\n🎉 ALL PIPELINES FINISHED!")

if __name__ == "__main__":
    main()