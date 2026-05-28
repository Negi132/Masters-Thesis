"""
TRANSFORMER COMPLETION EXPERIMENTS
====================================
Two-part experiment runner:

PART A: Fill in missing Transformer DK1 baseline experiments
   - Each horizon needs all 13 feature sets for complete Plot 8 coverage
   - Skips experiments that already exist in the CSV
   - Estimated remaining: ~125 experiments

PART B: Add Transformer DK2 experiments (24h, both targets)
   - Matches the existing LSTM DK2 spot check structure
   - Uses Best/Mean/Worst feature sets derived from DK1 baseline
   - Allows Transformer to replace LSTM as the NN representative in DK2 comparison

Run from /Code/ML_Pipeline/ directory:
    python run_transformer_completion.py

The script is resumable - safe to Ctrl+C and restart. Only mid-flight
experiments will be lost.

CSV column order safeguard: Uses evaluator.log_experiment() consistently
so the misalignment issue from Optuna won't repeat.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Setup path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import evaluator
from ML_Pipeline import model_trainer

# =====================================================================
# CONFIGURATION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = "experiment_results.csv"

# Part toggles - set to False to skip a part
RUN_PART_A_DK1_FILL = True   # Fill missing Transformer DK1 experiments
RUN_PART_B_DK2_RUN  = True   # Add Transformer DK2 experiments

# Skip experiments that already have results in the CSV
SKIP_EXISTING = True

# Only the Transformer model
MODEL_NAME = "Transformer"

# All 13 feature sets
ALL_EXPERIMENTS = [
    "Exp1_Weather_Only",
    "Exp2_Weather_WeatherLags_Only",
    "Exp3_Weather_Prices",
    "Exp4_Weather_WeatherLags_Prices",
    "Exp5_Weather_Grid",
    "Exp6_Weather_WeatherLags_Grid",
    "Exp7_Weather_Grid_Prices",
    "Exp8_Weather_WeatherLags_Grid_Prices",
    "Exp9_Weather_Grid_Gridlags",
    "Exp10_Weather_WeatherLags_Grid_Gridlags",
    "Exp11_Weather_Grid_Gridlags_Prices",
    "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices",
    "Exp13_Total_Information",
]

# Horizons per target
PRICE_HORIZONS = [0, 24, 48, 72, 96, 120, 144, 168]
DELTA_HORIZONS = [24, 48, 72, 96, 120, 144, 168]

# Map experiment names to feature groups
EXP_GROUPS = {
    "Exp1_Weather_Only":                              ["Weather", "Time"],
    "Exp2_Weather_WeatherLags_Only":                  ["Weather", "WeatherLags", "Time"],
    "Exp3_Weather_Prices":                            ["Weather", "Prices", "Time"],
    "Exp4_Weather_WeatherLags_Prices":                ["Weather", "WeatherLags", "Prices", "Time"],
    "Exp5_Weather_Grid":                              ["Weather", "Grid", "GridExchange", "Time"],
    "Exp6_Weather_WeatherLags_Grid":                  ["Weather", "WeatherLags", "Grid", "GridExchange", "Time"],
    "Exp7_Weather_Grid_Prices":                       ["Weather", "Grid", "GridExchange", "Prices", "Time"],
    "Exp8_Weather_WeatherLags_Grid_Prices":           ["Weather", "WeatherLags", "Grid", "GridExchange", "Prices", "Time"],
    "Exp9_Weather_Grid_Gridlags":                     ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"],
    "Exp10_Weather_WeatherLags_Grid_Gridlags":        ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"],
    "Exp11_Weather_Grid_Gridlags_Prices":             ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"],
    "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"],
    "Exp13_Total_Information":                        ["All_Features"],
}


# =====================================================================
# HELPERS
# =====================================================================
def print_section(title, subtitle=""):
    print("\n" + "=" * 70)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70)


def set_all_models_off():
    config.RUN_XGBOOST = config.RUN_LIGHTGBM = config.RUN_CATBOOST = False
    config.RUN_RANDOM_FOREST = config.RUN_LSTM = config.RUN_GRU = False
    config.RUN_TRANSFORMER = config.RUN_AUTOGLUON = False


def enable_transformer():
    set_all_models_off()
    config.RUN_TRANSFORMER = True


def run_experiment(name, groups, region, horizon, target):
    config.EXPERIMENT_NAME = name
    config.ACTIVE_GROUPS = groups
    config.REGION = region
    config.HORIZON = horizon
    config.TARGET_COL = f"TARGET_{target}_{horizon}h"
    print(f"\n  Running: {name}")
    print(f"  | Region: {region} | Horizon: {horizon}h | Target: {target}")
    print(f"  | Feature groups: {groups}")
    start = time.time()
    try:
        model_trainer.run_walk_forward_pipeline()
        print(f"  -> [SUCCESS] {(time.time()-start)/60:.1f} min")
        return True
    except Exception as e:
        print(f"  -> [ERROR] {e}")
        return False


def clean_exp_name(name):
    for tag in ['_0h','_24h','_48h','_72h','_96h','_120h','_144h','_168h']:
        if tag in str(name):
            return str(name).split(tag)[0]
    return str(name)


def load_existing_experiments(region_filter=None):
    """
    Returns a set of (region, target, horizon, base_exp) tuples that
    already exist as SUCCESS for the Transformer model.
    """
    if not os.path.exists(CSV_FILE):
        return set()
    
    try:
        df = pd.read_csv(CSV_FILE, sep=None, engine='python')
        df = df[df['Status'] == 'SUCCESS']
        df = df[df['Model'] == MODEL_NAME]
        
        if region_filter:
            df = df[df['Region'] == region_filter]
        
        # Exclude special experiments (MAELoss/Optuna/Midas/Pruned) for the baseline coverage check
        # These tagged experiments shouldn't satisfy the "missing baseline" requirement
        df = df[~df['Experiment'].str.contains(
            'MAELoss|Optuna|Midas|Pruned|FullWeek|Fullweek',
            case=False, na=False
        )]
        
        if df.empty:
            return set()
        
        df['Target_Type'] = df['Target'].astype(str).apply(
            lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
        df['Horizon'] = df['Target'].astype(str).apply(
            lambda x: int(x.split('_')[2].replace('h','')) if len(x.split('_')) > 2 else -1)
        df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
        
        existing = set()
        for _, row in df.iterrows():
            existing.add((
                row['Region'],
                row['Target_Type'],
                row['Horizon'],
                row['Base_Experiment']
            ))
        return existing
    except Exception as e:
        print(f"  [WARNING] Could not check existing experiments: {e}")
        return set()


def detect_best_mean_worst_features():
    """
    Identifies best/mean/worst performing feature sets for Transformer
    based on DK1 24h Price baseline. Used for DK2 experiments.
    """
    df = pd.read_csv(CSV_FILE, sep=None, engine='python')
    df = df[df['Status'] == 'SUCCESS'].copy()
    df = df[df['Model'] == MODEL_NAME]
    df = df[df['Region'] == 'DK1']
    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h','')) if len(x.split('_')) > 2 else -1)
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    
    # Filter to special-free, 24h, Price
    df = df[~df['Experiment'].str.contains(
        'MAELoss|Optuna|Midas|Pruned|FullWeek|Fullweek|DK2',
        case=False, na=False
    )]
    mask = (df['Target_Type'] == 'Price') & (df['Horizon'] == 24)
    df_24 = df[mask]
    
    if df_24.empty:
        # Fallback: hardcode sensible defaults
        print("  [WARNING] No Transformer 24h Price baseline data found.")
        print("  Falling back to defaults: Best=Exp13, Mean=Exp3, Worst=Exp2")
        return "Exp13_Total_Information", "Exp3_Weather_Prices", "Exp2_Weather_WeatherLags_Only"
    
    exp_mae = df_24.groupby('Base_Experiment')['MAE'].mean().reset_index()
    best_feat  = exp_mae.loc[exp_mae['MAE'].idxmin(), 'Base_Experiment']
    worst_feat = exp_mae.loc[exp_mae['MAE'].idxmax(), 'Base_Experiment']
    mean_mae   = exp_mae['MAE'].mean()
    exp_mae['diff'] = abs(exp_mae['MAE'] - mean_mae)
    mean_feat  = exp_mae.sort_values('diff').iloc[0]['Base_Experiment']
    
    print(f"  Best:  {best_feat}")
    print(f"  Mean:  {mean_feat}")
    print(f"  Worst: {worst_feat}")
    
    return best_feat, mean_feat, worst_feat


# =====================================================================
# PART A: FILL MISSING DK1 TRANSFORMER EXPERIMENTS
# =====================================================================
def run_part_a():
    print_section("PART A: FILL MISSING TRANSFORMER DK1 EXPERIMENTS",
                  "Goal: Get all 13 feature sets for every horizon")
    
    # Build the complete target list
    targets_to_run = []
    
    # Price horizons
    for h in PRICE_HORIZONS:
        for exp in ALL_EXPERIMENTS:
            targets_to_run.append(('DK1', 'Price', h, exp))
    
    # Delta horizons
    for h in DELTA_HORIZONS:
        for exp in ALL_EXPERIMENTS:
            targets_to_run.append(('DK1', 'Delta', h, exp))
    
    total_needed = len(targets_to_run)
    print(f"\n  Total Transformer DK1 experiments needed: {total_needed}")
    
    # Check what already exists
    if SKIP_EXISTING:
        existing = load_existing_experiments(region_filter='DK1')
        print(f"  Already in CSV (SUCCESS): {len(existing)}")
        targets_to_run = [t for t in targets_to_run if t not in existing]
        print(f"  Remaining to run: {len(targets_to_run)}")
    
    if not targets_to_run:
        print("\n  Part A complete - no missing experiments!")
        return 0, 0
    
    # Run them
    enable_transformer()
    config.USE_PRUNING_ENGINE = False
    
    completed = 0
    failed = 0
    start_time = time.time()
    
    for idx, (region, target, horizon, base_exp) in enumerate(targets_to_run, 1):
        groups = EXP_GROUPS.get(base_exp)
        if groups is None:
            print(f"  [SKIP] Unknown experiment: {base_exp}")
            failed += 1
            continue
        
        elapsed = (time.time() - start_time) / 60
        avg = elapsed / completed if completed > 0 else 0
        eta = (len(targets_to_run) - idx) * avg
        
        print(f"\n[A: {idx}/{len(targets_to_run)}] {target} {horizon}h | {base_exp}")
        print(f"  Elapsed: {elapsed:.1f} min | ETA: {eta:.1f} min")
        
        exp_name = f"{base_exp}_{horizon}h_{target}"
        success = run_experiment(
            name=exp_name, groups=groups,
            region=region, horizon=horizon, target=target
        )
        
        if success:
            completed += 1
        else:
            failed += 1
    
    return completed, failed


# =====================================================================
# PART B: TRANSFORMER DK2 EXPERIMENTS
# =====================================================================
def run_part_b():
    print_section("PART B: TRANSFORMER DK2 EXPERIMENTS",
                  "Goal: Add Transformer to DK2 spot check (24h, both targets)")
    
    # Check DK2 master matrix exists
    dk2_check = config.ML_DATA_DIR / "Master_Matrix_DK2_Horizon0h.csv"
    if not dk2_check.exists():
        print(f"\n  [PART B SKIPPED] DK2 master matrix not found at {dk2_check}")
        print(f"  Run build_region_master('DK2') first to generate DK2 data.")
        return 0, 0
    
    # Determine feature sets from Transformer DK1 24h Price baseline
    print(f"\n  Detecting Best/Mean/Worst feature sets...")
    best_feat, mean_feat, worst_feat = detect_best_mean_worst_features()
    
    feature_sets = [
        ("Best",  best_feat,  EXP_GROUPS.get(best_feat, ["All_Features"])),
        ("Mean",  mean_feat,  EXP_GROUPS.get(mean_feat, ["All_Features"])),
        ("Worst", worst_feat, EXP_GROUPS.get(worst_feat, ["All_Features"])),
    ]
    
    # Build the target list: 3 feature sets × 2 targets = 6 experiments
    targets_to_run = []
    for label, feat_name, _ in feature_sets:
        for target in ["Price", "Delta"]:
            targets_to_run.append(('DK2', target, 24, feat_name, label))
    
    total_needed = len(targets_to_run)
    print(f"\n  Total Transformer DK2 experiments needed: {total_needed}")
    
    # Check what already exists
    if SKIP_EXISTING:
        existing = load_existing_experiments(region_filter='DK2')
        # For DK2, also strip the "_DK2" suffix from existing exp names if present
        existing_dk2 = set()
        for (region, target, horizon, base_exp) in existing:
            # The DK2 naming convention adds _DK2_ to the experiment name
            # We need to handle this when checking
            cleaned = base_exp.split('_DK2')[0]
            existing_dk2.add((region, target, horizon, cleaned))
        
        before = len(targets_to_run)
        targets_to_run = [
            t for t in targets_to_run
            if ('DK2', t[1], t[2], t[3]) not in existing_dk2
        ]
        skipped = before - len(targets_to_run)
        if skipped > 0:
            print(f"  Skipping {skipped} already-complete DK2 experiments")
        print(f"  Remaining to run: {len(targets_to_run)}")
    
    if not targets_to_run:
        print("\n  Part B complete - no missing experiments!")
        return 0, 0
    
    # Run them
    enable_transformer()
    config.USE_PRUNING_ENGINE = False
    
    completed = 0
    failed = 0
    start_time = time.time()
    
    for idx, (region, target, horizon, feat_name, label) in enumerate(targets_to_run, 1):
        groups = EXP_GROUPS.get(feat_name)
        
        elapsed = (time.time() - start_time) / 60
        avg = elapsed / completed if completed > 0 else 0
        eta = (len(targets_to_run) - idx) * avg
        
        print(f"\n[B: {idx}/{len(targets_to_run)}] DK2 {target} {horizon}h | {label} ({feat_name})")
        print(f"  Elapsed: {elapsed:.1f} min | ETA: {eta:.1f} min")
        
        # Naming matches the existing DK2 convention: {exp}_DK2_24h_{target}
        exp_name = f"{feat_name}_DK2_{horizon}h_{target}"
        success = run_experiment(
            name=exp_name, groups=groups,
            region=region, horizon=horizon, target=target
        )
        
        if success:
            completed += 1
        else:
            failed += 1
    
    # Reset region to DK1 to avoid affecting subsequent runs
    config.REGION = "DK1"
    
    return completed, failed


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    print_section("TRANSFORMER COMPLETION EXPERIMENTS",
                  "DK1 baseline fill + DK2 spot check")
    
    overall_start = time.time()
    total_completed = 0
    total_failed = 0
    
    if RUN_PART_A_DK1_FILL:
        a_done, a_fail = run_part_a()
        total_completed += a_done
        total_failed += a_fail
    else:
        print("\n  [PART A SKIPPED] (RUN_PART_A_DK1_FILL = False)")
    
    if RUN_PART_B_DK2_RUN:
        b_done, b_fail = run_part_b()
        total_completed += b_done
        total_failed += b_fail
    else:
        print("\n  [PART B SKIPPED] (RUN_PART_B_DK2_RUN = False)")
    
    total_time = (time.time() - overall_start) / 60
    
    print_section("ALL DONE")
    print(f"  Completed: {total_completed} experiments")
    if total_failed > 0:
        print(f"  Failed:    {total_failed} experiments")
    print(f"  Total time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    print(f"\n  Next steps:")
    print(f"    1. python cleanDuplicateCSV.py")
    print(f"    2. cd ../Plotting && python master_plotter.py")
    print("=" * 70)
