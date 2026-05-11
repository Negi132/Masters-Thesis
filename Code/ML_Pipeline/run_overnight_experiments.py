import time
import sys
import os

# Add parent directory to path to allow imports from ML_Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import model_trainer

def main():
    print("==================================================")
    print("      MASSIVE EXPERIMENT SUITE INITIALIZED")
    print("==================================================")
    
    # =========================================================
    # RESUME CONTROLLER
    # Change this number to start from a specific experiment!
    # (e.g., If 3 finished, set this to 4 to resume)
    # =========================================================
    START_EXPERIMENT = 1
    
    # 1. Define the BASE templates (No horizons or targets here)
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

    # 2. DYNAMICALLY GENERATE THE QUEUE
    experiment_queue = []
    
    # Targets: Price, then Delta
    for target in ["Price", "Delta"]:
        # Horizons: 0 up to 168, in steps of 24
        for horizon in range(0, 169, 24):
            for base in base_experiments:
                # Create a dynamic name so your CSV clearly shows what is happening
                # e.g., "Exp1_Weather_Only_24h_Delta"
                exp_name = f"{base['base_name']}_{horizon}h_{target}"
                
                experiment_queue.append({
                    "name": exp_name,
                    "groups": base["groups"],
                    "region": "DK1", # Hardcoded for now, can easily be looped later
                    "horizon": horizon,
                    "target_type": target
                })

    print(f"Successfully generated {len(experiment_queue)} experiments.")
    print("="*50)
    
    total_start_time = time.time()

    # 3. RUN THE QUEUE
    for i, exp in enumerate(experiment_queue, 1):

        # SKIP ALREADY COMPLETED EXPERIMENTS
        if i < START_EXPERIMENT:
            continue
        
        print(f"\n[{i}/{len(experiment_queue)}] INITIALIZING: {exp['name']}")
        
        config.EXPERIMENT_NAME = exp["name"]
        config.ACTIVE_GROUPS = exp["groups"]
        config.REGION = exp.get("region", "DK1")
        config.HORIZON = exp.get("horizon", 0)
        target_type = exp.get("target_type", "Price")
        
        config.TARGET_COL = f"TARGET_{target_type}_{config.HORIZON}h"
        
        print(f"  | Region:  {config.REGION}")
        print(f"  | Horizon: {config.HORIZON}h")
        print(f"  | Target:  {config.TARGET_COL}")
        print(f"  | Groups:  {config.ACTIVE_GROUPS}")
        print("-" * 50)
        
        try:
            model_trainer.run_walk_forward_pipeline()
            print(f"  -> [SUCCESS] {exp['name']} completed successfully.")
        except Exception as e:
            print(f"  -> [FATAL ERROR] {exp['name']} failed to execute:")
            print(f"     {e}")

    total_time = (time.time() - total_start_time) / 3600
    print("\n" + "="*50)
    print(f"SUITE FINISHED in {total_time:.2f} hours.")
    print("="*50)

if __name__ == "__main__":
    main()