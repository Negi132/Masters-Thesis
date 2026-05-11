import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import model_trainer

def main():
    print("==================================================")
    print("      NEURAL NETWORK VARIANCE SUITE INITIALIZED")
    print("==================================================")
    
    # Testing all 13 feature sets for BOTH targets at a chosen horizon
    TARGETS = ["Price", "Delta"]
    HORIZON = 24  # You can change this if you want to map variance for 48h, 72h, etc.
    
    base_experiments = [
        {"base_name": "Exp1_Weather_Only", "groups": ["Weather", "Time"]},
        {"base_name": "Exp2_Weather_WeatherLags_Only", "groups": ["Weather", "WeatherLags", "Time"]},
        {"base_name": "Exp3_Weather_Prices", "groups": ["Weather", "Prices", "Time"]},
        {"base_name": "Exp4_Weather_WeatherLags_Prices", "groups": ["Weather", "WeatherLags", "Prices", "Time"]},
        {"base_name": "Exp5_Weather_Grid", "groups": ["Weather", "Grid", "GridExchange", "Time"]},
        {"base_name": "Exp6_Weather_WeatherLags_Grid", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "Time"]},
        {"base_name": "Exp7_Weather_Grid_Prices", "groups": ["Weather", "Grid", "GridExchange", "Prices", "Time"]},
        {"base_name": "Exp8_Weather_WeatherLags_Grid_Prices", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "Prices", "Time"]},
        {"base_name": "Exp9_Weather_Grid_Gridlags", "groups": ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"]},
        {"base_name": "Exp10_Weather_WeatherLags_Grid_Gridlags", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"]},
        {"base_name": "Exp11_Weather_Grid_Gridlags_Prices", "groups": ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"]},
        {"base_name": "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices", "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"]},
        {"base_name": "Exp13_Total_Information", "groups": ["All_Features"]}
    ]

    print(f"Executing {len(base_experiments) * len(TARGETS)} total experiments.")
    print("="*50)
    
    total_start_time = time.time()

    for target in TARGETS:
        for i, exp in enumerate(base_experiments, 1):
            full_name = f"{exp['base_name']}_{HORIZON}h_{target}"
            
            print(f"\n[{target} | {i}/{len(base_experiments)}] INITIALIZING: {full_name}")
            
            config.EXPERIMENT_NAME = full_name
            config.ACTIVE_GROUPS = exp["groups"]
            config.REGION = "DK1"
            config.HORIZON = HORIZON
            config.TARGET_COL = f"TARGET_{target}_{HORIZON}h"
            
            try:
                model_trainer.run_walk_forward_pipeline()
                print(f"  -> [SUCCESS] {full_name} completed.")
            except Exception as e:
                print(f"  -> [ERROR] {full_name} failed: {e}")

    total_time = (time.time() - total_start_time) / 3600
    print("\n" + "="*50)
    print(f"VARIANCE SUITE FINISHED in {total_time:.2f} hours.")
    print("="*50)

if __name__ == "__main__":
    main()