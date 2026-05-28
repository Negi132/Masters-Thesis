"""
Stage 3 Rerun - Midas Substitution with Fixes
==============================================
Re-runs the Midas substitution stage with three fixes applied:

  Fix 1: NaN imputation in Midas weather columns.
         The original Midas JSON contains nulls (especially in
         avg_sun_last1h_glob and avg_cloud_cover). Trees tolerate NaN;
         NN scalers do not. The first run produced garbage NN output
         (identical WMAPE, MDA near 0.5%) because of this. The Midas
         matrix builder is now run with imputation applied to the
         Midas columns mirroring the DMI pipeline's behaviour.

  Fix 2: Exp13 / All_Features handled correctly.
         The first run skipped any model whose best feature set was
         Exp13 because "you cannot substitute Weather group when the
         feature set is All_Features." But the Midas matrix ALREADY
         has DMI weather stripped and Midas weather added, so running
         Exp13 (= all columns in the matrix) on the Midas matrix
         IS the substitution. This script removes the skip and
         iterates Exp13 like any other experiment.

  Fix 3: Idempotent re-run.
         Deletes any existing _MidasSub and _MidasRangeDMI rows from
         experiment_results.csv (and the clean version) before
         re-running, so the rerun produces a clean, deduplicated
         result set rather than a mix of new and broken old rows.

Stage 1 (GRU-tanh) and Stage 2 (Optuna walk-forward) are NOT touched.
Only Stage 3 (Midas substitution + DMI control) is re-run.
"""

import sys
import os
import copy
import shutil
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(THIS_DIR))
sys.path.append(THIS_DIR)

from ML_Pipeline import config
from ML_Pipeline import data_loader
from ML_Pipeline import model_trainer
from ML_Pipeline import evaluator


# =====================================================================
# CONFIG
# =====================================================================
CSV_FILE = "experiment_results.csv"
CSV_CLEAN_FILE = "experiment_results_clean.csv"

MIDAS_DATA_DIR = Path("../Data_Engineering/Data/DMI/weather-data")
MIDAS_FILE_DK1 = MIDAS_DATA_DIR / "weather-dk1.json"

MIDAS_RAW_COLUMNS = [
    "avg_temp_dry", "avg_humidity", "avg_cloud_cover", "avg_wind_dir",
    "avg_wind_speed", "avg_radia_glob_past1h", "avg_sun_last1h_glob",
]
MIDAS_RENAME = {col: f"midas_{col[4:]}" if col.startswith("avg_") else f"midas_{col}"
                for col in MIDAS_RAW_COLUMNS}
UNIVERSAL_LAG_HOURS = 24
HORIZONS = [0, 24, 48, 72, 96, 120, 144, 168]

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


# =====================================================================
# MIDAS MATRIX BUILDER (with NaN imputation)
# =====================================================================
def get_midas_column_groups():
    midas_weather = list(MIDAS_RENAME.values())
    midas_weather_lags = [f"{col}_lag_{UNIVERSAL_LAG_HOURS}h" for col in midas_weather]
    return {
        "MidasWeather":     midas_weather,
        "MidasWeatherLags": midas_weather_lags,
    }


def load_and_impute_midas(json_path=MIDAS_FILE_DK1):
    """Loads Midas JSON, imputes NaN values, returns a tidy DataFrame.

    Imputation mirrors the DMI preprocessing pipeline (Script 9):
      1. Interpolation for gaps up to 12 hours.
      2. Forward then backward fill for any remaining edge NaN, up to 12h.
      3. Column mean for any survivors.
    """
    print(f"  Loading Midas JSON: {json_path}")
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Midas JSON not found at {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"  Raw rows: {len(df):,}")

    df['HourUTC'] = pd.to_datetime(df['datetime'], utc=True).dt.floor('h')
    df = df.drop(columns=['datetime'])
    df = df.sort_values('HourUTC').drop_duplicates(subset='HourUTC', keep='last').reset_index(drop=True)

    df = df[['HourUTC'] + MIDAS_RAW_COLUMNS]
    df = df.rename(columns=MIDAS_RENAME)

    for col in MIDAS_RENAME.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # NaN inventory BEFORE imputation
    nan_before = df.isna().sum()
    nan_cols = [c for c in df.columns if c != 'HourUTC' and nan_before[c] > 0]
    if nan_cols:
        print(f"  NaN values before imputation:")
        for c in nan_cols:
            print(f"    {c}: {nan_before[c]:,}")

    # Stage 1: interpolation (limit 12)
    for col in MIDAS_RENAME.values():
        df[col] = df[col].interpolate(method='linear', limit=12, limit_direction='both')

    # Stage 2: forward fill then backward fill (limit 12 each)
    for col in MIDAS_RENAME.values():
        df[col] = df[col].ffill(limit=12).bfill(limit=12)

    # Stage 3: column mean for any survivors
    survivors = df[list(MIDAS_RENAME.values())].isna().sum()
    survivors = survivors[survivors > 0]
    if not survivors.empty:
        print(f"  Survivors after fill (using column mean):")
        for col, n in survivors.items():
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"    {col}: {n:,} filled with mean = {mean_val:.4f}")

    # Sanity
    remaining = df[list(MIDAS_RENAME.values())].isna().sum().sum()
    if remaining > 0:
        print(f"  [WARN] {remaining} NaN values remain after all imputation passes!")
    else:
        print(f"  [OK] All Midas weather NaN values imputed.")

    print(f"  Date range: {df['HourUTC'].min()} -> {df['HourUTC'].max()}")
    return df


def add_midas_lags(df_midas):
    midas_cols = [c for c in df_midas.columns if c != 'HourUTC']
    for col in midas_cols:
        df_midas[f"{col}_lag_{UNIVERSAL_LAG_HOURS}h"] = df_midas[col].shift(UNIVERSAL_LAG_HOURS)
    return df_midas


def identify_dmi_weather_columns():
    weather = config.COL_GROUPS.get("Weather", [])
    weather_lags = config.COL_GROUPS.get("WeatherLags", [])
    return set(weather) | set(weather_lags)


def build_midas_matrix_for_horizon(horizon, df_midas_with_lags):
    """Builds the per-horizon Midas master matrix. ALWAYS overwrites
    any existing file (no skip-if-exists check) so the rerun produces
    fresh matrices with the imputed Midas data."""
    src_path = config.ML_DATA_DIR / f"Master_Matrix_DK1_Horizon{horizon}h.csv"
    dst_path = config.ML_DATA_DIR / f"Master_Matrix_DK1_Midas_Horizon{horizon}h.csv"

    if not src_path.exists():
        print(f"  [WARN] DMI source matrix missing: {src_path}")
        return

    print(f"\n  Building horizon {horizon}h:")
    df = pd.read_csv(src_path)
    df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True)

    dmi_weather_cols = identify_dmi_weather_columns()
    cols_to_drop = [c for c in df.columns if c in dmi_weather_cols]
    df = df.drop(columns=cols_to_drop)

    imputed_weather = [c for c in df.columns
                       if c.endswith('_imputed') and any(w in c for w in
                                                          ['temp', 'wind', 'sun', 'humid',
                                                           'cloud', 'precip', 'pressure',
                                                           'radia', 'leaf'])]
    if imputed_weather:
        df = df.drop(columns=imputed_weather)

    df_merged = pd.merge(df, df_midas_with_lags, on='HourUTC', how='inner')

    lag_cols = [c for c in df_merged.columns if c.endswith(f'_lag_{UNIVERSAL_LAG_HOURS}h')
                and c.startswith('midas_')]
    if lag_cols:
        before = len(df_merged)
        df_merged = df_merged.dropna(subset=lag_cols).reset_index(drop=True)
        print(f"    Dropped {before - len(df_merged):,} rows with NaN Midas lags at start")

    # Final NaN check on Midas columns specifically
    midas_cols = [c for c in df_merged.columns if c.startswith('midas_')]
    nan_in_midas = df_merged[midas_cols].isna().sum().sum()
    if nan_in_midas > 0:
        print(f"    [WARN] {nan_in_midas} NaN values in Midas columns of final matrix!")
    else:
        print(f"    [OK] No NaN in Midas columns ({len(midas_cols)} cols, {len(df_merged):,} rows)")

    df_merged['HourUTC'] = df_merged['HourUTC'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    df_merged.to_csv(dst_path, index=False)
    print(f"    Saved: {dst_path.name}")


def rebuild_all_midas_matrices():
    print("=" * 70)
    print("  REBUILDING MIDAS MATRICES WITH IMPUTATION")
    print("=" * 70)
    df_midas = load_and_impute_midas()
    df_midas = add_midas_lags(df_midas)
    for h in HORIZONS:
        build_midas_matrix_for_horizon(h, df_midas)
    print("\n  Midas matrix rebuild complete.")
    return df_midas


def get_midas_date_range(df_midas_with_lags):
    lag_cols = [c for c in df_midas_with_lags.columns
                if c.endswith(f'_lag_{UNIVERSAL_LAG_HOURS}h')]
    df = df_midas_with_lags.dropna(subset=lag_cols)
    return df['HourUTC'].min(), df['HourUTC'].max()


# =====================================================================
# CSV CLEANUP - remove broken old MidasSub/MidasRangeDMI rows
# =====================================================================
def clean_old_midas_rows():
    print("=" * 70)
    print("  CLEANING OLD BROKEN MIDAS ROWS FROM CSV")
    print("=" * 70)
    for csv_path in [CSV_FILE, CSV_CLEAN_FILE]:
        if not os.path.exists(csv_path):
            print(f"  Skip: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path, sep=None, engine='python')
        before = len(df)
        mask = df['Experiment'].astype(str).str.contains(
            'MidasSub|MidasRangeDMI', case=False, na=False)
        removed = mask.sum()
        df = df[~mask]
        df.to_csv(csv_path, index=False)
        print(f"  {csv_path}: removed {removed:,} rows  ({before:,} -> {len(df):,})")


def clean_old_midas_pkls():
    log_dir = Path("Experiment_Logs")
    if not log_dir.exists():
        print(f"  No Experiment_Logs directory to clean")
        return
    print(f"  Cleaning old Midas pkls from {log_dir}/")
    removed = 0
    for pkl in log_dir.glob("*MidasSub*.pkl"):
        pkl.unlink()
        removed += 1
    for pkl in log_dir.glob("*MidasRangeDMI*.pkl"):
        pkl.unlink()
        removed += 1
    print(f"  Removed {removed} Midas pkl files")


# =====================================================================
# CONFIG SNAPSHOT/RESTORE
# =====================================================================
def snapshot_config():
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


def set_all_models_off():
    config.RUN_XGBOOST = config.RUN_LIGHTGBM = config.RUN_CATBOOST = False
    config.RUN_RANDOM_FOREST = config.RUN_LSTM = config.RUN_GRU = False
    config.RUN_TRANSFORMER = config.RUN_AUTOGLUON = False


def enable_single_model(model_name):
    set_all_models_off()
    m = {
        "XGBoost": "RUN_XGBOOST",     "LightGBM": "RUN_LIGHTGBM",
        "CatBoost": "RUN_CATBOOST",   "RandomForest": "RUN_RANDOM_FOREST",
        "LSTM": "RUN_LSTM",           "GRU": "RUN_GRU",
        "Transformer": "RUN_TRANSFORMER", "AutoGluon": "RUN_AUTOGLUON",
    }
    if model_name in m:
        setattr(config, m[model_name], True)


# =====================================================================
# BEST FEATURE SET DISCOVERY
# =====================================================================
def best_feature_set_per_model(target_type, csv_path=CSV_FILE):
    """Returns {model_name: base_experiment_name} for the lowest mean
    MAE baseline feature set at 24h horizon per target_type."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path, sep=None, engine='python')
    df = df[df['Status'] == 'SUCCESS'].copy()
    df = df[~df['Experiment'].astype(str).str.contains(
        'Pruned|FullWeek|Fullweek|Midas|Optuna|MAELoss|DK2|GRUtanh|Naive|'
        'medium_quality|best_quality',
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
        grouped = sub.groupby('Base_Experiment')['MAE'].mean().reset_index()
        best_exp = grouped.loc[grouped['MAE'].idxmin(), 'Base_Experiment']
        best[model] = best_exp
    return best


# =====================================================================
# EXPERIMENT RUNNERS
# =====================================================================
def run_midas_substitution_single(target, model_name, base_exp, groups):
    exp_name = f"{base_exp}_24h_{target}_MidasSub"
    print(f"    [{model_name}] Midas substitution: {exp_name}")

    enable_single_model(model_name)
    config.REGION = "DK1_Midas"
    config.TARGET_COL = f"TARGET_{target}_24h"
    config.EXPERIMENT_NAME = exp_name
    config.ACTIVE_GROUPS = groups

    try:
        model_trainer.run_walk_forward_pipeline()
    except Exception as e:
        print(f"      [ERROR] {exp_name}: {e}")


def run_dmi_date_matched_single(target, model_name, base_exp, groups,
                                start, end):
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
# STAGE 3 RERUN
# =====================================================================
def run_stage_3():
    print("=" * 70)
    print("  STAGE 3 RERUN: Midas substitution with NaN fix and Exp13 handling")
    print("=" * 70)

    # 1. Rebuild matrices
    df_midas = rebuild_all_midas_matrices()

    # 2. Register Midas groups in config
    midas_groups = get_midas_column_groups()
    config.COL_GROUPS["MidasWeather"] = midas_groups["MidasWeather"]
    config.COL_GROUPS["MidasWeatherLags"] = midas_groups["MidasWeatherLags"]

    # 3. Get Midas date range
    midas_start, midas_end = get_midas_date_range(df_midas)
    print(f"\n  Midas usable date range: {midas_start} -> {midas_end}")

    config.USE_PRUNING_ENGINE = False
    config.HORIZON = 24

    for target in ["Price", "Delta"]:
        print(f"\n  Target: {target}")
        try:
            best_map = best_feature_set_per_model(target)
        except Exception as e:
            print(f"    [SKIP] cannot derive best feature sets: {e}")
            continue

        for model_name, base_exp in best_map.items():
            groups = BASE_EXPERIMENTS_MAP[base_exp]

            # FIX 2: Handle All_Features (Exp13) by running it directly
            # against the Midas matrix. The matrix already has DMI weather
            # stripped and Midas weather added, so "All_Features" against
            # it = the correct Midas substitution. No group rewriting needed.
            if "All_Features" in groups:
                midas_subbed_groups = ["All_Features"]
                print(f"    [{model_name}] Best set is Exp13/All_Features - "
                      f"running directly against Midas matrix (substitution "
                      f"happens automatically via the matrix's column composition).")
            else:
                # Substitute Weather/WeatherLags groups with Midas equivalents
                midas_subbed_groups = []
                for g in groups:
                    if g == "Weather":
                        midas_subbed_groups.append("MidasWeather")
                    elif g == "WeatherLags":
                        midas_subbed_groups.append("MidasWeatherLags")
                    else:
                        midas_subbed_groups.append(g)

            run_midas_substitution_single(
                target, model_name, base_exp, midas_subbed_groups
            )
            run_dmi_date_matched_single(
                target, model_name, base_exp, groups,
                midas_start, midas_end
            )


def main():
    print("=" * 70)
    print("  STAGE 3 RERUN ORCHESTRATOR")
    print("=" * 70)
    print("  This script re-runs ONLY the Midas substitution stage.")
    print("  Stages 1 (GRU-tanh) and 2 (Optuna-WF) are not affected.")
    print()

    overall_start = time.time()
    snap = snapshot_config()

    try:
        clean_old_midas_rows()
        clean_old_midas_pkls()
        run_stage_3()
    except Exception as e:
        print(f"\n  [FATAL] Stage 3 rerun crashed: {e}")
        import traceback; traceback.print_exc()
    finally:
        restore_config(snap)

    elapsed = (time.time() - overall_start) / 60
    print("\n" + "=" * 70)
    print(f"  STAGE 3 RERUN COMPLETE in {elapsed:.1f} minutes")
    print("=" * 70)
    print("  Next steps:")
    print("    1. python cleanDuplicateCSV.py")
    print("    2. Regenerate plots via master_plotter")
    print("=" * 70)


if __name__ == "__main__":
    main()
