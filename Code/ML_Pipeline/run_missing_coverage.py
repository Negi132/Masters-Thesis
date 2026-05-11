"""
MISSING COVERAGE TRAINER
=========================
Reads missing_coverage_queue.csv produced by analyze_missing_coverage.py
and trains the missing experiments. Only runs the models that are actually
missing for each experiment/horizon/target combination — avoids rerunning
models that already have data.
"""

import sys
import os
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import model_trainer

QUEUE_FILE = "missing_coverage_queue.csv"

# These are the NN models — we set them separately since coverage gaps
# are expected to be NN-only at non-24h horizons
TREE_MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
NN_MODELS   = ["LSTM", "GRU", "Transformer", "AutoGluon"]

def set_models_for(models_needed):
    """Enable only the specific models needed for this run."""
    config.RUN_XGBOOST       = "XGBoost"      in models_needed
    config.RUN_LIGHTGBM      = "LightGBM"     in models_needed
    config.RUN_CATBOOST      = "CatBoost"     in models_needed
    config.RUN_RANDOM_FOREST = "RandomForest" in models_needed
    config.RUN_LSTM          = "LSTM"         in models_needed
    config.RUN_GRU           = "GRU"          in models_needed
    config.RUN_TRANSFORMER   = "Transformer"  in models_needed
    config.RUN_AUTOGLUON     = "AutoGluon"    in models_needed

def main():
    print("=" * 60)
    print("  MISSING COVERAGE TRAINER")
    print("=" * 60)

    if not os.path.exists(QUEUE_FILE):
        print(f"[ERROR] {QUEUE_FILE} not found.")
        print("  Run analyze_missing_coverage.py first.")
        return

    queue_df = pd.read_csv(QUEUE_FILE)
    print(f"  Loaded {len(queue_df)} gap entries from queue.\n")

    # Load current CSV to check what models already exist per experiment
    try:
        results_df = pd.read_csv("experiment_results.csv", sep=None, engine='python')
        results_df = results_df[results_df['Status'] == 'SUCCESS'].copy()
        results_df['Target_Type'] = results_df['Target'].astype(str).apply(
            lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
        results_df['Horizon'] = results_df['Target'].astype(str).apply(
            lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)

        def clean_exp(name):
            base = str(name)
            for tag in ['_0h','_24h','_48h','_72h','_96h','_120h','_144h','_168h']:
                if tag in base:
                    base = base.split(tag)[0]
                    break
            return base

        results_df['Base_Experiment'] = results_df['Experiment'].apply(clean_exp)
        results_df['Version'] = results_df['Experiment'].apply(
            lambda x: 'Pruned' if 'Pruned' in str(x) else
                      ('FullWeek' if ('FullWeek' in str(x) or 'Fullweek' in str(x)) else 'Baseline'))
        baseline_df = results_df[results_df['Version'] == 'Baseline']
    except Exception as e:
        print(f"[WARNING] Could not read results CSV to check existing models: {e}")
        baseline_df = pd.DataFrame()

    # Group queue by experiment × horizon × target to batch model runs
    grouped = queue_df.groupby(['experiment', 'horizon', 'target'])

    total_groups = len(grouped)
    total_start  = time.time()
    config.USE_PRUNING_ENGINE = False  # Baseline runs — no pruning

    for i, ((exp_name, horizon, target), group_df) in enumerate(grouped, 1):
        # Parse which models are needed from the queue directly
        raw_models = group_df['needed_by_models'].iloc[0]
        if isinstance(raw_models, str):
            import ast
            try:
                models_from_queue = ast.literal_eval(raw_models)
            except Exception:
                models_from_queue = TREE_MODELS + NN_MODELS
        else:
            models_from_queue = TREE_MODELS + NN_MODELS

        # Double-check against CSV to avoid rerunning anything already present
        if not baseline_df.empty:
            already_present = set(
                baseline_df[
                    (baseline_df['Base_Experiment'] == exp_name) &
                    (baseline_df['Horizon'] == horizon) &
                    (baseline_df['Target_Type'] == target)
                ]['Model'].unique()
            )
        else:
            already_present = set()

        models_needed = [m for m in models_from_queue if m not in already_present]

        if not models_needed:
            print(f"[{i}/{total_groups}] SKIP {exp_name} | {horizon}h | {target} — already complete")
            continue

        # Get groups from the first row (all rows in group share the same groups value)
        raw_groups = group_df['groups'].iloc[0]
        # Parse the groups string back to a list (stored as string in CSV)
        if isinstance(raw_groups, str):
            import ast
            try:
                active_groups = ast.literal_eval(raw_groups)
            except Exception:
                active_groups = ["All_Features"]
        else:
            active_groups = ["All_Features"]

        full_name = f"{exp_name}_{horizon}h_{target}"

        print(f"\n[{i}/{total_groups}] INITIALIZING: {full_name}")
        print(f"  | Models needed: {models_needed}")
        print(f"  | Groups:        {active_groups}")
        print("-" * 60)

        config.EXPERIMENT_NAME = full_name
        config.ACTIVE_GROUPS   = active_groups
        config.REGION          = "DK1"
        config.HORIZON         = horizon
        config.TARGET_COL      = f"TARGET_{target}_{horizon}h"

        set_models_for(models_needed)

        start_t = time.time()
        try:
            model_trainer.run_walk_forward_pipeline()
            elapsed = (time.time() - start_t) / 60
            print(f"  -> [SUCCESS] {full_name} completed in {elapsed:.1f} min.")
        except Exception as e:
            print(f"  -> [ERROR] {full_name} failed: {e}")

    total_time = (time.time() - total_start) / 3600
    print("\n" + "=" * 60)
    print(f"  COVERAGE TRAINING COMPLETE in {total_time:.2f} hours.")
    print(f"  Re-run analyze_missing_coverage.py to verify coverage.")
    print("=" * 60)

if __name__ == "__main__":
    main()