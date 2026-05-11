import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import model_trainer

# =====================================================================
# TARGETED BASELINE RECOVERY
# Reruns only the two experiments with missing tree model pkl data.
# NNs are left untouched by the merge fix in model_trainer.py
# =====================================================================

# Safety check - ensure pruning is off for clean baseline runs
config.USE_PRUNING_ENGINE = False

# Ensure only tree models run
config.RUN_XGBOOST      = True
config.RUN_LIGHTGBM     = True
config.RUN_CATBOOST     = True
config.RUN_RANDOM_FOREST = True
config.RUN_LSTM         = False
config.RUN_GRU          = False
config.RUN_TRANSFORMER  = False
config.RUN_AUTOGLUON    = False

RECOVERY_QUEUE = [
    {
        "name":   "Exp13_Total_Information_24h_Price",
        "groups": ["All_Features"],
        "region": "DK1",
        "horizon": 24,
        "target": "Price"
    },
    {
        "name":   "Exp8_Weather_WeatherLags_Grid_Prices_24h_Delta",
        "groups": ["Weather", "WeatherLags", "Grid", "GridExchange", "Prices", "Time"],
        "region": "DK1",
        "horizon": 24,
        "target": "Delta"
    }
]

def main():
    print("=" * 60)
    print("  BASELINE RECOVERY - Tree Models Only")
    print(f"  Running {len(RECOVERY_QUEUE)} targeted experiments")
    print("=" * 60)

    for i, exp in enumerate(RECOVERY_QUEUE, 1):
        print(f"\n[{i}/{len(RECOVERY_QUEUE)}] {exp['name']}")

        config.EXPERIMENT_NAME  = exp["name"]
        config.ACTIVE_GROUPS    = exp["groups"]
        config.REGION           = exp["region"]
        config.HORIZON          = exp["horizon"]
        config.TARGET_COL       = f"TARGET_{exp['target']}_{exp['horizon']}h"

        print(f"  | Groups:  {config.ACTIVE_GROUPS}")
        print(f"  | Target:  {config.TARGET_COL}")
        print("-" * 60)

        try:
            model_trainer.run_walk_forward_pipeline()
            print(f"  -> [SUCCESS] {exp['name']} complete.")
        except Exception as e:
            print(f"  -> [ERROR] {exp['name']} failed: {e}")

    print("\n" + "=" * 60)
    print("  RECOVERY COMPLETE.")
    print("  Run validate_plotter_data.py again to confirm all OK.")
    print("=" * 60)

if __name__ == "__main__":
    main()