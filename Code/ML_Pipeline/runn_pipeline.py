"""
MASTER PIPELINE RUNNER
======================
Runs the full pipeline in the correct order:
  Stage 1: recover_missing_baselines  (fix missing tree pkl files)
  Stage 2: 16_permutation_importance  (generate pruned_features JSON files)
  Stage 3: 17_run_tree_full_horizon_pruned  (fast - trees with pruning)
  Stage 4: 15_run_nn_full_horizon     (slow - leave overnight)

Config is updated correctly between each step.
"""

import sys
import os
import importlib.util

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# RESUME CONTROL - Set to False if Stage 1 already completed
# =====================================================================
RUN_STAGE_1 = False
RUN_STAGE_2 = False
RUN_STAGE_3 = True
RUN_STAGE_4 = True

# =====================================================================
# HELPERS
# =====================================================================

def print_stage(n, title, subtitle=""):
    print("\n" + "=" * 60)
    print(f"  STAGE {n}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 60)

def load_module(filename):
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def set_config_trees_no_pruning():
    """Stage 1 & 2: tree models only, pruning off."""
    config.USE_PRUNING_ENGINE = False
    config.RUN_XGBOOST        = True
    config.RUN_LIGHTGBM       = True
    config.RUN_CATBOOST       = True
    config.RUN_RANDOM_FOREST  = True
    config.RUN_LSTM           = False
    config.RUN_GRU            = False
    config.RUN_TRANSFORMER    = False
    config.RUN_AUTOGLUON      = False

def set_config_trees_pruned():
    """Stage 3: tree models only, pruning on."""
    config.USE_PRUNING_ENGINE = True
    config.RUN_XGBOOST        = True
    config.RUN_LIGHTGBM       = True
    config.RUN_CATBOOST       = True
    config.RUN_RANDOM_FOREST  = True
    config.RUN_LSTM           = False
    config.RUN_GRU            = False
    config.RUN_TRANSFORMER    = False
    config.RUN_AUTOGLUON      = False

def set_config_nn_pruned():
    """Stage 4: NN models only, pruning on."""
    config.USE_PRUNING_ENGINE = True
    config.RUN_XGBOOST        = False
    config.RUN_LIGHTGBM       = False
    config.RUN_CATBOOST       = False
    config.RUN_RANDOM_FOREST  = False
    config.RUN_LSTM           = True
    config.RUN_GRU            = True
    config.RUN_TRANSFORMER    = True
    config.RUN_AUTOGLUON      = True

# =====================================================================
# STAGE 1: RECOVER MISSING BASELINES
# =====================================================================
if RUN_STAGE_1:
    print_stage(1, "BASELINE RECOVERY", "Reruns 2 missing tree experiments (fast)")
    set_config_trees_no_pruning()

    try:
        recovery = load_module("recover_missing_baselines.py")
        recovery.main()
        print("\n  [STAGE 1 COMPLETE]")
    except Exception as e:
        print(f"\n  [STAGE 1 FAILED] {e}")
        print("  Fix the error above before continuing.")
        sys.exit(1)

# =====================================================================
# STAGE 2: PERMUTATION IMPORTANCE (Script 16)
# =====================================================================
if RUN_STAGE_2:
    print_stage(2, "PERMUTATION IMPORTANCE (Script 16)",
                "Generates pruned_features_Price.json and pruned_features_Delta.json")

    # Script 16 manages its own model toggles internally.
    # Pruning must be OFF so it trains on full unfiltered features.
    config.USE_PRUNING_ENGINE = False

    try:
        perm = load_module("16_permutation_importance.py")
        perm.execute_pruning_engine(target_type="Price")
        perm.execute_pruning_engine(target_type="Delta")
        print("\n  [STAGE 2 COMPLETE]")
    except Exception as e:
        print(f"\n  [STAGE 2 FAILED] {e}")
        print("  Fix the error above before continuing.")
        sys.exit(1)

# =====================================================================
# STAGE 3: TREE FULL HORIZON PRUNED (Script 17)
# =====================================================================
if RUN_STAGE_3:
    print_stage(3, "TREE FULL HORIZON PRUNED (Script 17)",
                "All tree models across all horizons with pruned features (fast)")
    set_config_trees_pruned()

    try:
        tree_runner = load_module("17_run_tree_full_horizon_pruned.py")
        tree_runner.main()
        print("\n  [STAGE 3 COMPLETE]")
    except Exception as e:
        print(f"\n  [STAGE 3 FAILED] {e}")
        print("  Fix the error above before continuing.")
        sys.exit(1)

# =====================================================================
# STAGE 4: NN FULL HORIZON (Script 15) - LEAVE OVERNIGHT
# =====================================================================
if RUN_STAGE_4:
    print_stage(4, "NN FULL HORIZON (Script 15)",
                "All NN models across all horizons with pruned features (SLOW - leave overnight)")
    set_config_nn_pruned()

    try:
        nn_runner = load_module("15_run_nn_full_horizon.py")
        nn_runner.main()
        print("\n  [STAGE 4 COMPLETE]")
    except Exception as e:
        print(f"\n  [STAGE 4 FAILED] {e}")
        sys.exit(1)

# =====================================================================
# DONE
# =====================================================================
print("\n" + "=" * 60)
print("  ALL STAGES COMPLETE")
print("  Run 18_master_plotter.py to generate all plots.")
print("=" * 60)