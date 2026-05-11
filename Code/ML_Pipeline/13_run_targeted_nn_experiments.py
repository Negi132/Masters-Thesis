import time
import sys
import os

# Add parent directory to path to allow imports from ML_Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import model_trainer

def main():
    print("==================================================")
    print("      TARGETED NEURAL NETWORK SUITE INITIALIZED")
    print("==================================================")
    
    # The exact winners extracted from the summary report
    targeted_queue = [
        # --- PRICE TARGET WINNERS ---
        {"base": "Exp11_Weather_Grid_Gridlags_Prices", "groups": ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"], "h": 0, "tgt": "Price"},
        {"base": "Exp13_Total_Information", "groups": ["All_Features"], "h": 24, "tgt": "Price"},
        {"base": "Exp13_Total_Information", "groups": ["All_Features"], "h": 48, "tgt": "Price"},
        {"base": "Exp13_Total_Information", "groups": ["All_Features"], "h": 72, "tgt": "Price"},
        {"base": "Exp13_Total_Information", "groups": ["All_Features"], "h": 96, "tgt": "Price"},
        {"base": "Exp3_Weather_Prices", "groups": ["Weather", "Prices", "Time"], "h": 120, "tgt": "Price"},
        {"base": "Exp4_Weather_WeatherLags_Prices", "groups": ["Weather", "WeatherLags", "Prices", "Time"], "h": 144, "tgt": "Price"},
        {"base": "Exp3_Weather_Prices", "groups": ["Weather", "Prices", "Time"], "h": 168, "tgt": "Price"},
        
        # --- DELTA TARGET WINNERS ---
        {"base": "Exp8_Weather_WeatherLags_Grid_Prices", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "Prices", "Time"], "h": 24, "tgt": "Delta"},
        {"base": "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"], "h": 48, "tgt": "Delta"},
        {"base": "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"], "h": 72, "tgt": "Delta"},
        {"base": "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"], "h": 96, "tgt": "Delta"},
        {"base": "Exp11_Weather_Grid_Gridlags_Prices", "groups": ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"], "h": 120, "tgt": "Delta"},
        {"base": "Exp4_Weather_WeatherLags_Prices", "groups": ["Weather", "WeatherLags", "Prices", "Time"], "h": 144, "tgt": "Delta"},
        {"base": "Exp4_Weather_WeatherLags_Prices", "groups": ["Weather", "WeatherLags", "Prices", "Time"], "h": 168, "tgt": "Delta"}
    ]

    print(f"Successfully loaded {len(targeted_queue)} optimal experiments.")
    print("="*50)
    
    total_start_time = time.time()

    for i, exp in enumerate(targeted_queue, 1):
        # Format the name just like the CSV expects
        full_name = f"{exp['base']}_{exp['h']}h_{exp['tgt']}"
        
        print(f"\n[{i}/{len(targeted_queue)}] INITIALIZING: {full_name}")
        
        config.EXPERIMENT_NAME = full_name
        config.ACTIVE_GROUPS = exp["groups"]
        config.REGION = "DK1"
        config.HORIZON = exp["h"]
        config.TARGET_COL = f"TARGET_{exp['tgt']}_{exp['h']}h"
        
        try:
            model_trainer.run_walk_forward_pipeline()
            print(f"  -> [SUCCESS] {full_name} completed successfully.")
        except Exception as e:
            print(f"  -> [FATAL ERROR] {full_name} failed to execute:")
            print(f"     {e}")

    total_time = (time.time() - total_start_time) / 3600
    print("\n" + "="*50)
    print(f"SUITE FINISHED in {total_time:.2f} hours.")
    print("="*50)

if __name__ == "__main__":
    main()