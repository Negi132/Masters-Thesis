"""
TARGETED FOLLOW-UP EXPERIMENTS ORCHESTRATOR
============================================
Runs three follow-up investigations identified during paper review:

  Stage 1: GRU-tanh full horizon (Price target only)
           Investigates whether the GRU instability at Price horizons
           0h, 48h, 72h, 96h is caused by ReLU on the recurrent layer.

  Stage 2: Optuna with walk-forward objective (fixes broken tuning)
           Original Optuna optimised against a single 30-day window,
           which caused Delta predictions to degrade by ~40%. This
           re-runs with a proper walk-forward objective.

  Stage 3: Midas weather substitution (replaces DMI weather)
           Each model's best feature set at 24h has its Weather and
           WeatherLags groups swapped out for Midas equivalents. Both
           Midas and a date-range-matched DMI baseline are run for
           fair comparison.

Each stage is independently toggleable. They share configuration
restoration so that running any subset leaves config in a clean state.

Naming conventions for CSV/pkl outputs:
  GRU-tanh:  <Exp>_<H>h_Price_GRUtanh                  Model="GRU_tanh"
  Optuna:    <Exp>_24h_<Target>_OptunaWF               Model=<tree model>
  Midas:     <Exp>_24h_<Target>_MidasSub               Model=<model>
  DMI ctrl:  <Exp>_24h_<Target>_MidasRangeDMI          Model=<model>
"""

import sys
import os
import time
import copy
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Allow imports from the ML_Pipeline package and local helper modules
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(THIS_DIR))
sys.path.append(THIS_DIR)

from ML_Pipeline import config
from ML_Pipeline import data_loader
from ML_Pipeline import model_trainer
from ML_Pipeline import evaluator


# =====================================================================
# STAGE TOGGLES - Set to False to skip a stage
# =====================================================================
RUN_STAGE_1_GRU_TANH       = True
RUN_STAGE_2_OPTUNA_WF      = True
RUN_STAGE_3_MIDAS_SUB      = True

# Optuna trial count per (model, feature_set, target) combination
OPTUNA_N_TRIALS = 30
OPTUNA_N_FOLDS = 3   # Walk-forward folds inside the objective

# Force rebuild of Midas master matrices even if they exist
MIDAS_FORCE_REBUILD = False

# =====================================================================
# CSV PATH (for "best feature set" discovery)
# =====================================================================
CSV_FILE = "experiment_results.csv"


# =====================================================================
# COMMON CONFIG MAPS
# =====================================================================
# Same mapping used in 15_run_nn_full_horizon.py and 17_run_tree_full_horizon_pruned.py
BASE_EXPERIMENTS_MAP = {
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

ALL_MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest",
              "LSTM", "GRU", "Transformer", "AutoGluon"]
TREE_MODELS = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]


# =====================================================================
# HELPERS
# =====================================================================
def print_stage(n, title, subtitle=""):
    print("\n" + "=" * 70)
    print(f"  STAGE {n}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 70)


def set_all_models_off():
    config.RUN_XGBOOST = config.RUN_LIGHTGBM = config.RUN_CATBOOST = False
    config.RUN_RANDOM_FOREST = config.RUN_LSTM = config.RUN_GRU = False
    config.RUN_TRANSFORMER = config.RUN_AUTOGLUON = False


def enable_single_model(model_name):
    """Match the existing weekend-script convention."""
    set_all_models_off()
    model_map = {
        "XGBoost": "RUN_XGBOOST",
        "LightGBM": "RUN_LIGHTGBM",
        "CatBoost": "RUN_CATBOOST",
        "RandomForest": "RUN_RANDOM_FOREST",
        "LSTM": "RUN_LSTM",
        "GRU": "RUN_GRU",
        "Transformer": "RUN_TRANSFORMER",
        "AutoGluon": "RUN_AUTOGLUON",
    }
    if model_name in model_map:
        setattr(config, model_map[model_name], True)


def snapshot_config():
    """Save a snapshot of relevant config fields that stages might mutate."""
    return {
        'REGION':              config.REGION,
        'HORIZON':             config.HORIZON,
        'TARGET_COL':          config.TARGET_COL,
        'EXPERIMENT_NAME':     config.EXPERIMENT_NAME,
        'ACTIVE_GROUPS':       list(config.ACTIVE_GROUPS),
        'USE_PRUNING_ENGINE':  config.USE_PRUNING_ENGINE,
        'COL_GROUPS':          copy.deepcopy(config.COL_GROUPS),
        'RUN_XGBOOST':         config.RUN_XGBOOST,
        'RUN_LIGHTGBM':        config.RUN_LIGHTGBM,
        'RUN_CATBOOST':        config.RUN_CATBOOST,
        'RUN_RANDOM_FOREST':   config.RUN_RANDOM_FOREST,
        'RUN_LSTM':            config.RUN_LSTM,
        'RUN_GRU':             config.RUN_GRU,
        'RUN_TRANSFORMER':     config.RUN_TRANSFORMER,
        'RUN_AUTOGLUON':       config.RUN_AUTOGLUON,
    }


def restore_config(snap):
    for k, v in snap.items():
        setattr(config, k, v)


# =====================================================================
# IDENTIFY BEST FEATURE SET PER MODEL AT 24H (used by Midas stage)
# =====================================================================
def best_feature_set_per_model(target_type, csv_path=CSV_FILE):
    """
    Returns {model_name: base_experiment_name} for the best (lowest mean
    MAE) baseline feature set at 24h horizon per target_type.

    Reads experiment_results.csv directly so this picks up any new data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found - cannot pick best feature sets.")

    df = pd.read_csv(csv_path, sep=None, engine='python')
    df = df[df['Status'] == 'SUCCESS'].copy()
    # Exclude experiment variants we don't want to derive "best from"
    df = df[~df['Experiment'].str.contains(
        'Pruned|FullWeek|Fullweek|Midas|Optuna|MAELoss|DK2|GRUtanh|Naive',
        case=False, na=False)]

    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)

    def clean(name):
        for tag in ['_0h', '_24h', '_48h', '_72h', '_96h', '_120h', '_144h', '_168h']:
            if tag in str(name):
                return str(name).split(tag)[0]
        return str(name)
    df['Base_Experiment'] = df['Experiment'].apply(clean)

    df = df[(df['Horizon'] == 24) & (df['Target_Type'] == target_type)]
    best = {}
    for model in df['Model'].unique():
        sub = df[df['Model'] == model]
        if sub.empty:
            continue
        # Mean MAE per base experiment, lowest wins
        grouped = sub.groupby('Base_Experiment')['MAE'].mean().reset_index()
        best_exp = grouped.loc[grouped['MAE'].idxmin(), 'Base_Experiment']
        best[model] = best_exp
    return best


# =====================================================================
# STAGE 1: GRU-tanh FULL HORIZON (Price only)
# =====================================================================
def run_stage_1_gru_tanh():
    print_stage(1, "GRU-tanh full horizon",
                "Activation: tanh (default) instead of relu. Price target only.")

    # Identify the same Best/Mean/Worst feature sets used in
    # 15_run_nn_full_horizon.py so the comparison is direct.
    try:
        df = pd.read_csv(CSV_FILE, sep=None, engine='python')
    except Exception as e:
        print(f"  [ERROR] cannot read {CSV_FILE}: {e}")
        return

    df = df[df['Status'] == 'SUCCESS'].copy()
    df = df[~df['Experiment'].str.contains(
        'Pruned|FullWeek|Fullweek|Midas|Optuna|MAELoss|DK2|GRUtanh|Naive',
        case=False, na=False)]
    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)

    def clean(name):
        for tag in ['_0h', '_24h', '_48h', '_72h', '_96h', '_120h', '_144h', '_168h']:
            if tag in str(name):
                return str(name).split(tag)[0]
        return str(name)
    df['Base_Experiment'] = df['Experiment'].apply(clean)

    # Use GRU's own ranking at 24h horizon to pick best/mean/worst
    df_gru = df[(df['Model'] == 'GRU') & (df['Horizon'] == 24) &
                (df['Target_Type'] == 'Price')]
    if df_gru.empty:
        print("  [SKIP] No GRU Price 24h data to derive best/mean/worst from.")
        return

    grouped = df_gru.groupby('Base_Experiment')['MAE'].mean().reset_index()
    best_exp = grouped.loc[grouped['MAE'].idxmin(), 'Base_Experiment']
    worst_exp = grouped.loc[grouped['MAE'].idxmax(), 'Base_Experiment']
    mean_target = grouped['MAE'].mean()
    grouped['diff'] = (grouped['MAE'] - mean_target).abs()
    mean_exp = grouped.loc[grouped['diff'].idxmin(), 'Base_Experiment']

    feature_sets = list({best_exp, mean_exp, worst_exp})
    print(f"  Feature sets to test: {feature_sets}")
    print(f"  Horizons: 0, 24, 48, 72, 96, 120, 144, 168")
    print(f"  Total experiments: {len(feature_sets) * 8}")

    # Monkey-patch model_trainer.get_models so it returns our tanh-GRU
    # instead of the default GRU. We restore it after the stage.
    from gru_tanh_wrapper import KerasGRUtanhWrapper
    original_get_models = model_trainer.get_models

    def patched_get_models():
        return {"GRU_tanh": KerasGRUtanhWrapper(epochs=50, batch_size=64)}

    model_trainer.get_models = patched_get_models

    config.USE_PRUNING_ENGINE = False
    config.REGION = "DK1"

    try:
        for horizon in [0, 24, 48, 72, 96, 120, 144, 168]:
            for base in feature_sets:
                exp_name = f"{base}_{horizon}h_Price_GRUtanh"
                config.EXPERIMENT_NAME = exp_name
                config.ACTIVE_GROUPS = BASE_EXPERIMENTS_MAP[base]
                config.HORIZON = horizon
                config.TARGET_COL = f"TARGET_Price_{horizon}h"

                print(f"\n  Running {exp_name}")
                try:
                    model_trainer.run_walk_forward_pipeline()
                except Exception as e:
                    print(f"    [ERROR] {exp_name}: {e}")
    finally:
        # Always restore original get_models, even if a run errored
        model_trainer.get_models = original_get_models
        print("\n  [STAGE 1] Restored original get_models.")


# =====================================================================
# STAGE 2: OPTUNA WITH WALK-FORWARD OBJECTIVE
# =====================================================================
def run_stage_2_optuna_walkforward():
    print_stage(2, "Optuna with walk-forward objective",
                f"{OPTUNA_N_TRIALS} trials, {OPTUNA_N_FOLDS}-fold walk-forward per trial")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [SKIP] optuna not installed. pip install optuna")
        return

    from optuna_walkforward import (
        build_walkforward_folds, walkforward_objective
    )

    # Discover best/mean/worst feature sets from CSV (per target)
    if not os.path.exists(CSV_FILE):
        print(f"  [SKIP] {CSV_FILE} not found - cannot derive feature sets.")
        return

    df = pd.read_csv(CSV_FILE, sep=None, engine='python')
    df = df[df['Status'] == 'SUCCESS'].copy()
    df = df[~df['Experiment'].str.contains(
        'Pruned|FullWeek|Fullweek|Midas|Optuna|MAELoss|DK2|GRUtanh|Naive',
        case=False, na=False)]
    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)
    def clean(name):
        for tag in ['_0h', '_24h', '_48h', '_72h', '_96h', '_120h', '_144h', '_168h']:
            if tag in str(name):
                return str(name).split(tag)[0]
        return str(name)
    df['Base_Experiment'] = df['Experiment'].apply(clean)

    feature_sets_per_target = {}
    for target in ["Price", "Delta"]:
        sub = df[(df['Target_Type'] == target) & (df['Horizon'] == 24) &
                 (df['Model'].isin(TREE_MODELS))]
        if sub.empty:
            print(f"  [WARN] No 24h tree data for {target} - skipping target.")
            continue
        # Mean over all four tree models so we pick a set the family agrees on
        grouped = sub.groupby('Base_Experiment')['MAE'].mean().reset_index()
        best = grouped.loc[grouped['MAE'].idxmin(), 'Base_Experiment']
        worst = grouped.loc[grouped['MAE'].idxmax(), 'Base_Experiment']
        grouped['diff'] = (grouped['MAE'] - grouped['MAE'].mean()).abs()
        mean = grouped.loc[grouped['diff'].idxmin(), 'Base_Experiment']
        feature_sets_per_target[target] = list({best, mean, worst})
        print(f"  {target}: Best={best}, Mean={mean}, Worst={worst}")

    config.USE_PRUNING_ENGINE = False
    config.REGION = "DK1"
    config.HORIZON = 24

    for target, feature_sets in feature_sets_per_target.items():
        for exp_name in feature_sets:
            groups = BASE_EXPERIMENTS_MAP[exp_name]
            config.ACTIVE_GROUPS = groups
            config.TARGET_COL = f"TARGET_{target}_24h"
            config.EXPERIMENT_NAME = f"{exp_name}_24h_{target}_OptunaWF"

            print(f"\n  Optuna-WF: {exp_name} | {target}")

            # Load data once for this experiment
            raw_df = data_loader.load_master_data()

            for model_type in TREE_MODELS:
                print(f"    [{model_type}]")
                df_filtered = data_loader.get_filtered_features(
                    raw_df, active_groups=groups, model_name=model_type
                )

                try:
                    folds = build_walkforward_folds(
                        df_filtered, target_col=config.TARGET_COL,
                        n_folds=OPTUNA_N_FOLDS,
                    )
                except ValueError as e:
                    print(f"      [SKIP] {e}")
                    continue

                study = optuna.create_study(direction='minimize')
                study.optimize(
                    lambda trial: walkforward_objective(trial, model_type, folds),
                    n_trials=OPTUNA_N_TRIALS,
                    show_progress_bar=False,
                )

                best = study.best_trial
                print(f"      Best MAE (mean over {OPTUNA_N_FOLDS} folds): {best.value:.4f}")

                # Log
                metrics = {
                    "Timestamp":      pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Experiment":     config.EXPERIMENT_NAME,
                    "Region":         "DK1",
                    "Target":         config.TARGET_COL,
                    "Feature_Mask":   str(groups),
                    "Status":         "SUCCESS",
                    "RMSE":           np.nan,
                    "MAE":            round(best.value, 4),
                    "R2":             np.nan,
                    "WMAPE":          np.nan,
                    "sMAPE":          np.nan,
                    "MDA":            np.nan,
                    "Train_Time_Sec": round(sum(t.duration.total_seconds()
                                                for t in study.trials), 2),
                    "Model":          model_type,
                }
                evaluator.log_experiment(metrics)


# =====================================================================
# STAGE 3: MIDAS WEATHER SUBSTITUTION (+ date-matched DMI control)
# =====================================================================
def run_stage_3_midas_substitution():
    print_stage(3, "Midas weather substitution",
                "Per-model best feature set at 24h, with Weather and "
                "WeatherLags swapped for Midas equivalents.")

    # 1. Build Midas master matrices on disk if missing
    print("\n  [3.1] Building Midas master matrices...")
    from build_midas_matrix import (
        main as build_midas_main,
        get_midas_column_groups,
        load_midas_dataframe,
        add_midas_lags,
        get_midas_date_range,
    )
    build_midas_main(force=MIDAS_FORCE_REBUILD)

    # 2. Inject Midas groups into config.COL_GROUPS so feature filtering
    #    knows about them. Save originals to restore later.
    midas_groups = get_midas_column_groups()
    config.COL_GROUPS["MidasWeather"] = midas_groups["MidasWeather"]
    config.COL_GROUPS["MidasWeatherLags"] = midas_groups["MidasWeatherLags"]

    # 3. Determine the Midas date range so we can apply it to the DMI control
    df_midas = load_midas_dataframe()
    df_midas = add_midas_lags(df_midas)
    midas_start, midas_end = get_midas_date_range(df_midas)
    print(f"  Midas usable date range: {midas_start} -> {midas_end}")

    # 4. For each target, pick each model's best feature set and run two
    #    experiments: one with Midas substitution, one with DMI restricted
    #    to the Midas date range.
    config.USE_PRUNING_ENGINE = False
    config.HORIZON = 24

    for target in ["Price", "Delta"]:
        print(f"\n  [3.2] Target: {target}")
        try:
            best_map = best_feature_set_per_model(target)
        except Exception as e:
            print(f"    [SKIP] cannot derive best feature sets: {e}")
            continue

        for model_name, base_exp in best_map.items():
            groups = BASE_EXPERIMENTS_MAP[base_exp]

            # Substitute Weather/WeatherLags with Midas equivalents
            midas_subbed_groups = []
            for g in groups:
                if g == "Weather":
                    midas_subbed_groups.append("MidasWeather")
                elif g == "WeatherLags":
                    midas_subbed_groups.append("MidasWeatherLags")
                else:
                    midas_subbed_groups.append(g)

            # If the feature set was "All_Features" we cannot do a clean
            # substitution because that expands to literally all columns
            # including DMI weather. Skip with a note.
            if "All_Features" in groups:
                print(f"    [SKIP] {model_name}: best set is All_Features - "
                      f"substitution undefined.")
                continue

            # --------------- 3a. Midas-substituted run ---------------
            run_midas_substitution_single(
                target, model_name, base_exp, midas_subbed_groups
            )

            # --------------- 3b. DMI date-matched control run ---------------
            run_dmi_date_matched_single(
                target, model_name, base_exp, groups,
                midas_start, midas_end
            )


def run_midas_substitution_single(target, model_name, base_exp, groups):
    """One Midas substitution run for a single (target, model)."""
    exp_name = f"{base_exp}_24h_{target}_MidasSub"
    print(f"    [{model_name}] Midas substitution: {exp_name}")

    enable_single_model(model_name)
    config.REGION = "DK1_Midas"   # data_loader will look for *_DK1_Midas_*
    config.TARGET_COL = f"TARGET_{target}_24h"
    config.EXPERIMENT_NAME = exp_name
    config.ACTIVE_GROUPS = groups

    try:
        model_trainer.run_walk_forward_pipeline()
    except Exception as e:
        print(f"      [ERROR] {exp_name}: {e}")


def run_dmi_date_matched_single(target, model_name, base_exp, groups,
                                  start, end):
    """
    DMI control run on the same date range as Midas. We use the same
    feature set as the original (DMI weather) but restrict the master
    matrix to [start, end] before running walk-forward.

    Implementation note: rather than modifying data_loader, we wrap
    its load_master_data to apply the date filter. Restored on exit.
    """
    exp_name = f"{base_exp}_24h_{target}_MidasRangeDMI"
    print(f"    [{model_name}] DMI date-matched control: {exp_name}")

    enable_single_model(model_name)
    config.REGION = "DK1"
    config.TARGET_COL = f"TARGET_{target}_24h"
    config.EXPERIMENT_NAME = exp_name
    config.ACTIVE_GROUPS = groups

    original_loader = data_loader.load_master_data

    def date_filtered_loader():
        df = original_loader()
        # df['HourUTC'] is already datetime after load_master_data
        before = len(df)
        df = df[(df['HourUTC'] >= start) & (df['HourUTC'] <= end)].reset_index(drop=True)
        print(f"      Date-filtered DMI: {before:,} -> {len(df):,} rows")
        return df

    data_loader.load_master_data = date_filtered_loader
    try:
        model_trainer.run_walk_forward_pipeline()
    except Exception as e:
        print(f"      [ERROR] {exp_name}: {e}")
    finally:
        data_loader.load_master_data = original_loader


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("  TARGETED FOLLOW-UP EXPERIMENTS")
    print("=" * 70)
    print(f"  Stage 1 (GRU-tanh):     {RUN_STAGE_1_GRU_TANH}")
    print(f"  Stage 2 (Optuna-WF):    {RUN_STAGE_2_OPTUNA_WF}")
    print(f"  Stage 3 (Midas sub):    {RUN_STAGE_3_MIDAS_SUB}")
    print("=" * 70)

    overall_start = time.time()
    snap = snapshot_config()

    try:
        if RUN_STAGE_1_GRU_TANH:
            try:
                run_stage_1_gru_tanh()
            except Exception as e:
                print(f"\n  [STAGE 1 FAILED] {e}")
                import traceback; traceback.print_exc()
            finally:
                restore_config(snap)

        if RUN_STAGE_2_OPTUNA_WF:
            try:
                run_stage_2_optuna_walkforward()
            except Exception as e:
                print(f"\n  [STAGE 2 FAILED] {e}")
                import traceback; traceback.print_exc()
            finally:
                restore_config(snap)

        if RUN_STAGE_3_MIDAS_SUB:
            try:
                run_stage_3_midas_substitution()
            except Exception as e:
                print(f"\n  [STAGE 3 FAILED] {e}")
                import traceback; traceback.print_exc()
            finally:
                restore_config(snap)
    finally:
        restore_config(snap)

    elapsed = (time.time() - overall_start) / 60
    print("\n" + "=" * 70)
    print(f"  ALL STAGES COMPLETE in {elapsed:.1f} minutes")
    print("=" * 70)
    print("  Next steps:")
    print("    1. python cleanDuplicateCSV.py")
    print("    2. Regenerate plots via master_plotter")
    print("=" * 70)


if __name__ == "__main__":
    main()
