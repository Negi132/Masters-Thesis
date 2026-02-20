import subprocess
import sys
import time

# --- CONFIGURATION ---
# List your scripts here in the exact order you want them to run.
SCRIPTS_TO_RUN = [
    "1energyPricesReorderRenameTimesortingMergeSplit.py",
    "2removeBoringEmptyCoulmns.py",
]

def run_script(script_name):
    """
    Runs a python script and waits for it to finish.
    Returns True if successful, False otherwise.
    """
    print(f"--- Starting: {script_name} ---")
    start_time = time.time()
    
    try:
        # sys.executable ensures we use the same python interpreter (e.g. venv)
        result = subprocess.run([sys.executable, script_name], check=True)
        
        duration = time.time() - start_time
        print(f"✓ Finished: {script_name} (took {duration:.2f}s)\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{script_name}'")
        return False

def main():
    print(f"Starting pipeline with {len(SCRIPTS_TO_RUN)} scripts...\n")
    
    for script in SCRIPTS_TO_RUN:
        success = run_script(script)
        if not success:
            print("Pipeline stopped due to error.")
            sys.exit(1)
            
    print("All scripts executed successfully!")

if __name__ == "__main__":
    main()