import glob
import pickle

print("--- Checking files for Best (Exp13) and Worst (Exp1) ---")
for base_exp in ["Exp13_Total_Information", "Exp1_Weather_Only"]:
    pattern = f"Experiment_Logs/*{base_exp}*24h*Price*.pkl"
    files = glob.glob(pattern)
    
    print(f"\nSearching for: {pattern}")
    if not files:
        print("  -> NO FILES FOUND!")
        
    for f in files:
        # Only look at the Baseline files
        if "Pruned" not in f and "FullWeek" not in f and "Fullweek" not in f:
            print(f"  [BASELINE FILE FOUND] {f}")
            try:
                with open(f, 'rb') as pkl:
                    data = pickle.load(pkl)
                    print(f"  -> Models currently inside this file: {list(data.keys())}")
            except Exception as e:
                print(f"  -> Error reading file: {e}")