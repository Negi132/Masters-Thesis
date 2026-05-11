"""
WEEKEND EXPERIMENT PIPELINE (Fully Automated)
===============================================
Runs all supplementary experiments in the correct order:
  Stage 1: Naive Baseline          (no ML, instant)
  Stage 2: NN Loss Function Test   (MAE loss vs MSE, 24h, few hours)
  Stage 3: AutoGluon Quality Test  (medium_quality preset, 24h)
  Stage 4: Midas Energy Experiment (best/mean/worst, overnight)
  Stage 5: DK2 Spot Check          (best tree + best NN, 24h)
  Stage 6: Optuna Tree Tuning      (best/mean/worst, trees only)

All stages are fully automated. No manual code changes needed.
Best/mean/worst models and feature sets are detected from the CSV.
"""

import sys
import os
import time
import importlib.util
import pickle
import copy
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import data_loader
from ML_Pipeline import evaluator

# We import model_trainer components individually so we can patch them
from ML_Pipeline import model_trainer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE   = "experiment_results.csv"

# =====================================================================
# RESUME CONTROL
# =====================================================================
RUN_STAGE_1_NAIVE       = True
RUN_STAGE_2_LOSS_FN     = True
RUN_STAGE_3_AUTOGLUON   = True
RUN_STAGE_4_MIDAS       = True   # Requires Midas preprocessing first
RUN_STAGE_5_DK2         = True
RUN_STAGE_6_OPTUNA      = True

# =====================================================================
# SETTINGS
# =====================================================================
AUTOGLUON_QUALITY   = "medium_quality"
AUTOGLUON_TIME_LIMIT = 180  # seconds per fit call (3 min — your baseline used 6s)
OPTUNA_N_TRIALS     = 30

# =====================================================================
# HELPERS
# =====================================================================

def print_stage(n, title, subtitle=""):
    print("\n" + "=" * 60)
    print(f"  STAGE {n}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 60)

def set_all_models_off():
    config.RUN_XGBOOST = config.RUN_LIGHTGBM = config.RUN_CATBOOST = False
    config.RUN_RANDOM_FOREST = config.RUN_LSTM = config.RUN_GRU = False
    config.RUN_TRANSFORMER = config.RUN_AUTOGLUON = False

def set_trees_only():
    set_all_models_off()
    config.RUN_XGBOOST = config.RUN_LIGHTGBM = True
    config.RUN_CATBOOST = config.RUN_RANDOM_FOREST = True

def set_all_models_on():
    config.RUN_XGBOOST = config.RUN_LIGHTGBM = config.RUN_CATBOOST = True
    config.RUN_RANDOM_FOREST = config.RUN_LSTM = config.RUN_GRU = True
    config.RUN_TRANSFORMER = config.RUN_AUTOGLUON = True

def enable_single_model(model_name):
    set_all_models_off()
    model_map = {
        "XGBoost": "RUN_XGBOOST", "LightGBM": "RUN_LIGHTGBM",
        "CatBoost": "RUN_CATBOOST", "RandomForest": "RUN_RANDOM_FOREST",
        "LSTM": "RUN_LSTM", "GRU": "RUN_GRU",
        "Transformer": "RUN_TRANSFORMER", "AutoGluon": "RUN_AUTOGLUON",
    }
    if model_name in model_map:
        setattr(config, model_map[model_name], True)

def run_experiment(name, groups, region, horizon, target):
    config.EXPERIMENT_NAME = name
    config.ACTIVE_GROUPS   = groups
    config.REGION          = region
    config.HORIZON         = horizon
    config.TARGET_COL      = f"TARGET_{target}_{horizon}h"
    print(f"\n  Running: {name}")
    print(f"  | Region: {region} | Horizon: {horizon}h | Target: {target}")
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

def detect_best_models_and_features():
    """Reads the CSV and identifies best NN, best tree, and best/mean/worst features."""
    df = pd.read_csv(CSV_FILE, sep=None, engine='python')
    df = df[df['Status'] == 'SUCCESS'].copy()
    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h','')) if len(x.split('_')) > 2 else -1)
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    df = df[~df['Experiment'].str.contains('Pruned|FullWeek|Fullweek|Midas|Optuna|MAELoss|DK2',
                                            case=False, na=False)]

    # Filter to 24h Price Baseline
    mask = (df['Target_Type'] == 'Price') & (df['Horizon'] == 24)
    df_24 = df[mask]

    # Best tree model (lowest MAE among tree models)
    tree_names = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
    df_trees = df_24[df_24['Model'].isin(tree_names)]
    best_tree_row = df_trees.loc[df_trees['MAE'].idxmin()]
    best_tree = best_tree_row['Model']

    # Best NN model (lowest MAE among NN models)
    nn_names = ["LSTM", "GRU", "Transformer", "AutoGluon"]
    df_nns = df_24[df_24['Model'].isin(nn_names)]
    best_nn_row = df_nns.loc[df_nns['MAE'].idxmin()]
    best_nn = best_nn_row['Model']

    # Best/Mean/Worst feature sets (averaged across all models at 24h Price)
    exp_mae = df_24.groupby('Base_Experiment')['MAE'].mean().reset_index()
    best_feat  = exp_mae.loc[exp_mae['MAE'].idxmin(), 'Base_Experiment']
    worst_feat = exp_mae.loc[exp_mae['MAE'].idxmax(), 'Base_Experiment']
    mean_mae   = exp_mae['MAE'].mean()
    exp_mae['diff'] = abs(exp_mae['MAE'] - mean_mae)
    mean_feat  = exp_mae.sort_values('diff').iloc[0]['Base_Experiment']

    print(f"  Auto-detected from CSV:")
    print(f"    Best tree model:    {best_tree} (MAE: {best_tree_row['MAE']:.2f})")
    print(f"    Best NN model:      {best_nn} (MAE: {best_nn_row['MAE']:.2f})")
    print(f"    Best feature set:   {best_feat}")
    print(f"    Mean feature set:   {mean_feat}")
    print(f"    Worst feature set:  {worst_feat}")

    return best_tree, best_nn, best_feat, mean_feat, worst_feat

# Map experiment names to their column groups
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
# AUTO-DETECT BEST MODELS
# =====================================================================
print("=" * 60)
print("  WEEKEND EXPERIMENT PIPELINE")
print("  Detecting best models from baseline results...")
print("=" * 60)

BEST_TREE, BEST_NN, BEST_FEAT, MEAN_FEAT, WORST_FEAT = detect_best_models_and_features()

BEST_GROUPS  = EXP_GROUPS.get(BEST_FEAT,  ["All_Features"])
MEAN_GROUPS  = EXP_GROUPS.get(MEAN_FEAT,  ["All_Features"])
WORST_GROUPS = EXP_GROUPS.get(WORST_FEAT, ["All_Features"])


# =====================================================================
# STAGE 1: NAIVE BASELINE
# =====================================================================
if RUN_STAGE_1_NAIVE:
    print_stage(1, "NAIVE BASELINE",
                "Yesterday's price at same hour. No ML required.")
    try:
        master_path = config.ML_DATA_DIR / f"Master_Matrix_DK1_Horizon0h.csv"
        df_naive = pd.read_csv(master_path)

        print(f"\n  {'Horizon':<12} {'Naive MAE (EUR/MWh)':>20}")
        print(f"  {'-'*35}")
        for h in [24, 48, 72, 96, 120, 144, 168]:
            pred = df_naive['SpotPriceEUR'].shift(h).dropna()
            act  = df_naive['SpotPriceEUR'].iloc[h:].reset_index(drop=True)
            pred = pred.reset_index(drop=True)
            mae  = np.mean(np.abs(act - pred))
            print(f"  {h}h{'':<9} {mae:>20.2f}")

        print("\n  [STAGE 1 COMPLETE]")
    except Exception as e:
        print(f"\n  [STAGE 1 FAILED] {e}")


# =====================================================================
# STAGE 2: NN LOSS FUNCTION TEST (MAE loss)
# =====================================================================
if RUN_STAGE_2_LOSS_FN:
    print_stage(2, "NN LOSS FUNCTION TEST",
                f"Retraining {BEST_NN} with MAE loss (L1Loss) at 24h")

    config.USE_PRUNING_ENGINE = False

    # Monkey-patch the NN wrappers to use MAE loss instead of MSE
    _orig_rnn_fit = model_trainer.KerasRNNWrapper.fit
    _orig_tfm_fit = model_trainer.KerasTransformerWrapper.fit

    def _patched_rnn_fit(self, X, y, validation_data=None, verbose=0):
        if self.model is None:
            from keras.models import Sequential
            from keras.layers import LSTM, GRU, Dropout, Dense
            self.model = Sequential()
            input_shape = (X.shape[1], X.shape[2])
            if self.rnn_type == 'LSTM':
                self.model.add(LSTM(64, activation='relu', input_shape=input_shape))
            elif self.rnn_type == 'GRU':
                self.model.add(GRU(64, activation='relu', input_shape=input_shape))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(1))
            self.model.compile(optimizer='adam', loss='mae')  # <-- CHANGED
            print(f"  [LOSS FN] Compiled {self.rnn_type} with MAE loss")

        from keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks = [early_stop] if validation_data else []
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                                  validation_data=validation_data, verbose=verbose, callbacks=callbacks)
        self.train_loss = history.history.get('loss', [])
        self.val_loss = history.history.get('val_loss', [])

    def _patched_tfm_fit(self, X, y, validation_data=None, verbose=0):
        if self.model is None:
            from keras.layers import Input, MultiHeadAttention, Add, LayerNormalization, Dense, Flatten, Dropout
            from keras.models import Model
            input_shape = (X.shape[1], X.shape[2])
            inputs = Input(shape=input_shape)
            att = MultiHeadAttention(key_dim=32, num_heads=2)(inputs, inputs)
            att = Add()([inputs, att])
            att = LayerNormalization()(att)
            ff = Dense(32, activation='relu')(att)
            ff = Dense(X.shape[2])(ff)
            ff = Add()([att, ff])
            ff = LayerNormalization()(ff)
            x = Flatten()(ff)
            x = Dropout(0.2)(x)
            outputs = Dense(1)(x)
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer='adam', loss='mae')  # <-- CHANGED
            print(f"  [LOSS FN] Compiled Transformer with MAE loss")

        from keras.callbacks import EarlyStopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks = [early_stop] if validation_data else []
        history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                                  validation_data=validation_data, verbose=verbose, callbacks=callbacks)
        self.train_loss = history.history.get('loss', [])
        self.val_loss = history.history.get('val_loss', [])

    # Apply patches
    model_trainer.KerasRNNWrapper.fit = _patched_rnn_fit
    model_trainer.KerasTransformerWrapper.fit = _patched_tfm_fit

    # Run for the best NN only, at 24h, both targets
    enable_single_model(BEST_NN)
    for target in ["Price", "Delta"]:
        run_experiment(
            name    = f"{BEST_FEAT}_24h_{target}_MAELoss",
            groups  = BEST_GROUPS,
            region  = "DK1",
            horizon = 24,
            target  = target
        )

    # Restore original fit methods
    model_trainer.KerasRNNWrapper.fit = _orig_rnn_fit
    model_trainer.KerasTransformerWrapper.fit = _orig_tfm_fit

    print("\n  [STAGE 2 COMPLETE]")


# =====================================================================
# STAGE 3: AUTOGLUON QUALITY ESCALATION
# =====================================================================
if RUN_STAGE_3_AUTOGLUON:
    print_stage(3, "AUTOGLUON QUALITY ESCALATION",
                f"Testing '{AUTOGLUON_QUALITY}' preset at 24h")

    config.USE_PRUNING_ENGINE = False

    # Monkey-patch the AutoGluon wrapper to use a higher quality preset
    _orig_ag_fit = model_trainer.AutoGluonWrapper.fit

    def _patched_ag_fit(self, train_data, label, **kwargs):
        from autogluon.tabular import TabularPredictor
        self.predictor = TabularPredictor(label=label, path=self.path, verbosity=0).fit(
            train_data,
            time_limit=AUTOGLUON_TIME_LIMIT,
            presets=AUTOGLUON_QUALITY  # <-- CHANGED from 'medium_quality'
        )
        print(f"  [AUTOGLUON] Fitted with preset='{AUTOGLUON_QUALITY}', "
              f"time_limit={AUTOGLUON_TIME_LIMIT}s")

    model_trainer.AutoGluonWrapper.fit = _patched_ag_fit

    enable_single_model("AutoGluon")
    for target in ["Price", "Delta"]:
        run_experiment(
            name    = f"{BEST_FEAT}_24h_{target}_AutoGluon_{AUTOGLUON_QUALITY}",
            groups  = BEST_GROUPS,
            region  = "DK1",
            horizon = 24,
            target  = target
        )

    model_trainer.AutoGluonWrapper.fit = _orig_ag_fit
    print("\n  [STAGE 3 COMPLETE]")


# =====================================================================
# STAGE 4: MIDAS ENERGY EXPERIMENT
# =====================================================================
if RUN_STAGE_4_MIDAS:
    print_stage(4, "MIDAS ENERGY EXPERIMENT",
                "Preprocessing Midas data, then running best/mean/worst experiments")

    MIDAS_DATA_DIR = Path("../Data_Engineering/Data/DMI/weather-data")
    MIDAS_MATRIX_DIR = config.ML_DATA_DIR  # Same dir, different filename

    def preprocess_midas():
        """
        Loads Midas JSON weather data, merges it with existing master matrices,
        and saves augmented versions as separate files.
        """
        print("\n  --- Midas Preprocessing ---")

        for region in ["DK1", "DK2"]:
            midas_file = MIDAS_DATA_DIR / f"weather-{region.lower()}.json"

            if not midas_file.exists():
                print(f"  [WARNING] Midas file not found: {midas_file}")
                continue

            # Load Midas JSON
            midas_df = pd.read_json(midas_file)
            midas_df['HourUTC'] = pd.to_datetime(midas_df['datetime'], utc=True).dt.floor('h')
            midas_df = midas_df.drop(columns=['datetime'])
            midas_df = midas_df.sort_values('HourUTC').reset_index(drop=True)

            # Prefix all Midas columns to avoid name collisions
            midas_cols = [c for c in midas_df.columns if c != 'HourUTC']
            midas_df = midas_df.rename(columns={c: f"midas_{c}" for c in midas_cols})

            # Drop duplicates (some hours may appear twice)
            midas_df = midas_df.drop_duplicates(subset='HourUTC', keep='last')

            # Fill any missing Midas values (minor gaps)
            midas_numeric = [c for c in midas_df.columns if c != 'HourUTC']
            midas_df[midas_numeric] = midas_df[midas_numeric].interpolate(
                method='linear', limit=12, limit_direction='both')
            midas_df[midas_numeric] = midas_df[midas_numeric].ffill(limit=12).bfill(limit=12)

            # Also create 24h lags for Midas features
            for col in midas_numeric:
                midas_df[f"{col}_lag_24h"] = midas_df[col].shift(24)

            print(f"  [{region}] Midas data: {len(midas_df):,} hours, "
                  f"{len(midas_df.columns)-1} features (incl. lags)")
            print(f"  [{region}] Date range: {midas_df['HourUTC'].min()} to "
                  f"{midas_df['HourUTC'].max()}")

            # Merge with each existing master matrix
            for horizon in [0, 24, 48, 72, 96, 120, 144, 168]:
                master_file = config.ML_DATA_DIR / f"Master_Matrix_{region}_Horizon{horizon}h.csv"
                if not master_file.exists():
                    continue

                master_df = pd.read_csv(master_file)
                master_df['HourUTC'] = pd.to_datetime(master_df['HourUTC'], utc=True)

                # Inner merge — only keep hours where both master and Midas have data
                merged = pd.merge(master_df, midas_df, on='HourUTC', how='inner')

                # Drop rows with NaN from lag edges
                merged = merged.dropna()

                out_file = MIDAS_MATRIX_DIR / f"Master_Matrix_{region}_Horizon{horizon}h_Midas.csv"
                merged.to_csv(out_file, index=False)
                print(f"  [{region}] Saved Midas matrix: {out_file.name} "
                      f"({len(merged):,} rows, {len(merged.columns)} cols)")

        print("  --- Midas Preprocessing Complete ---\n")

    # Check if Midas matrices already exist, if not create them
    midas_check = MIDAS_MATRIX_DIR / "Master_Matrix_DK1_Horizon24h_Midas.csv"
    if not midas_check.exists():
        # Check if Midas source data exists
        midas_src = MIDAS_DATA_DIR / "weather-dk1.json"
        if not midas_src.exists():
            print(f"  [STAGE 4 SKIPPED] Midas source data not found at {MIDAS_DATA_DIR}")
            print(f"  Expected files: weather-dk1.json and weather-dk2.json")
        else:
            preprocess_midas()
    else:
        print("  Midas matrices already exist, skipping preprocessing.")

    # Now run experiments if Midas matrices exist
    midas_check = MIDAS_MATRIX_DIR / "Master_Matrix_DK1_Horizon24h_Midas.csv"
    if midas_check.exists():
        config.USE_PRUNING_ENGINE = False

        # Patch data_loader to load Midas version
        _orig_load = data_loader.load_master_data

        def _patched_load_midas():
            midas_path = config.ML_DATA_DIR / f"Master_Matrix_{config.REGION}_Horizon{config.HORIZON}h_Midas.csv"
            if not midas_path.exists():
                print(f"  [WARNING] Midas file not found for {config.REGION} "
                      f"Horizon {config.HORIZON}h, falling back to standard")
                return _orig_load()
            print(f"Loading {config.REGION} Midas Master Matrix for Horizon {config.HORIZON}h...")
            df = pd.read_csv(midas_path)
            df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True)
            df = df.sort_values('HourUTC').reset_index(drop=True)
            return df

        data_loader.load_master_data = _patched_load_midas

        # Phase 1: All models at 24h on best/mean/worst + Midas
        set_all_models_on()
        for exp_name, exp_groups in [(BEST_FEAT, BEST_GROUPS),
                                      (MEAN_FEAT, MEAN_GROUPS),
                                      (WORST_FEAT, WORST_GROUPS)]:
            for target in ["Price", "Delta"]:
                run_experiment(
                    name    = f"{exp_name}_Midas_24h_{target}",
                    groups  = ["All_Features"],
                    region  = "DK1",
                    horizon = 24,
                    target  = target
                )

        # Phase 2: Trees at all horizons with best feature set + Midas
        set_trees_only()
        for horizon in [0, 48, 72, 96, 120, 144, 168]:
            for target in ["Price", "Delta"]:
                if horizon == 0 and target == "Delta":
                    continue
                run_experiment(
                    name    = f"{BEST_FEAT}_Midas_{horizon}h_{target}",
                    groups  = ["All_Features"],
                    region  = "DK1",
                    horizon = horizon,
                    target  = target
                )

        # Restore original loader
        data_loader.load_master_data = _orig_load
        print("\n  [STAGE 4 COMPLETE]")


# =====================================================================
# STAGE 5: DK2 SPOT CHECK
# =====================================================================
if RUN_STAGE_5_DK2:
    print_stage(5, "DK2 SPOT CHECK",
                f"Best tree ({BEST_TREE}) + best NN ({BEST_NN}) on DK2 at 24h")

    config.USE_PRUNING_ENGINE = False

    # Check DK2 master matrix exists
    dk2_check = config.ML_DATA_DIR / "Master_Matrix_DK2_Horizon0h.csv"
    if not dk2_check.exists():
        print(f"  [STAGE 5 SKIPPED] DK2 master matrix not found at {dk2_check}")
        print(f"  Run script 9 with build_region_master('DK2') first.")
    else:
        for model_name in [BEST_TREE, BEST_NN]:
            enable_single_model(model_name)
            for exp_name, exp_groups in [(BEST_FEAT, BEST_GROUPS),
                                          (MEAN_FEAT, MEAN_GROUPS),
                                          (WORST_FEAT, WORST_GROUPS)]:
                for target in ["Price", "Delta"]:
                    run_experiment(
                        name    = f"{exp_name}_DK2_24h_{target}",
                        groups  = exp_groups,
                        region  = "DK2",
                        horizon = 24,
                        target  = target
                    )

        config.REGION = "DK1"  # Reset
        print("\n  [STAGE 5 COMPLETE]")


# =====================================================================
# STAGE 6: OPTUNA HYPERPARAMETER TUNING (Trees only)
# =====================================================================
if RUN_STAGE_6_OPTUNA:
    print_stage(6, "OPTUNA TREE TUNING",
                f"Trees only, best/mean/worst, {OPTUNA_N_TRIALS} trials each")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [STAGE 6 SKIPPED] Optuna not installed. Run: pip install optuna")
        RUN_STAGE_6_OPTUNA = False

    if RUN_STAGE_6_OPTUNA:
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error

        config.USE_PRUNING_ENGINE = False

        def optuna_objective(trial, model_type, X_train, y_train, X_val, y_val):
            """Optuna objective: returns MAE on validation set."""
            if model_type == "XGBoost":
                params = {
                    'max_depth':      trial.suggest_int('max_depth', 3, 10),
                    'learning_rate':  trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators':   trial.suggest_int('n_estimators', 50, 500),
                    'subsample':      trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'n_jobs': -1
                }
                model = XGBRegressor(**params)
            elif model_type == "LightGBM":
                params = {
                    'num_leaves':       trial.suggest_int('num_leaves', 20, 150),
                    'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'n_jobs': -1, 'verbose': -1
                }
                model = LGBMRegressor(**params)
            elif model_type == "CatBoost":
                params = {
                    'depth':          trial.suggest_int('depth', 4, 10),
                    'learning_rate':  trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'iterations':     trial.suggest_int('iterations', 50, 500),
                    'verbose': 0
                }
                model = CatBoostRegressor(**params)
            elif model_type == "RandomForest":
                params = {
                    'n_estimators':    trial.suggest_int('n_estimators', 50, 300),
                    'max_depth':       trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**params)

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return mean_absolute_error(y_val, preds)

        tree_models = ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]

        for exp_name, exp_groups in [(BEST_FEAT, BEST_GROUPS),
                                      (MEAN_FEAT, MEAN_GROUPS),
                                      (WORST_FEAT, WORST_GROUPS)]:
            for target in ["Price", "Delta"]:
                if target == "Delta" and 0 == 24:  # Never true, but keeps structure
                    continue

                config.EXPERIMENT_NAME = f"{exp_name}_24h_{target}_Optuna"
                config.ACTIVE_GROUPS   = exp_groups
                config.REGION          = "DK1"
                config.HORIZON         = 24
                config.TARGET_COL      = f"TARGET_{target}_24h"

                print(f"\n  Optuna tuning: {exp_name} | {target}")
                print("-" * 60)

                # Load data once for this experiment
                raw_df = data_loader.load_master_data()

                for model_type in tree_models:
                    df = data_loader.get_filtered_features(
                        raw_df, active_groups=exp_groups, model_name=model_type)

                    # Use last 2 years as train, last 30 days of that as validation
                    total = len(df)
                    val_size   = config.TEST_DAYS * 24
                    train_size = config.INITIAL_TRAIN_DAYS * 24

                    X_train, y_train, X_val, y_val = data_loader.get_train_test_split(
                        df, 0, total - val_size, total)

                    print(f"\n  [{model_type}] Running {OPTUNA_N_TRIALS} Optuna trials...")
                    study = optuna.create_study(direction='minimize')
                    study.optimize(
                        lambda trial: optuna_objective(trial, model_type, X_train, y_train, X_val, y_val),
                        n_trials=OPTUNA_N_TRIALS,
                        show_progress_bar=True
                    )

                    best = study.best_trial
                    print(f"  [{model_type}] Best MAE: {best.value:.4f}")
                    print(f"  [{model_type}] Best params: {best.params}")

                    # Log the best result to CSV
                    metrics = {
                        "Timestamp":    pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Experiment":   f"{exp_name}_24h_{target}_Optuna",
                        "Model":        model_type,
                        "Region":       "DK1",
                        "Target":       f"TARGET_{target}_24h",
                        "Feature_Mask": str(exp_groups),
                        "Status":       "SUCCESS",
                        "MAE":          round(best.value, 4),
                        "RMSE":         np.nan,
                        "R2":           np.nan,
                        "WMAPE":        np.nan,
                        "sMAPE":        np.nan,
                        "MDA":          np.nan,
                        "Train_Time_Sec": round(sum(t.duration.total_seconds()
                                                     for t in study.trials), 2)
                    }
                    evaluator.log_experiment(metrics)

        print("\n  [STAGE 6 COMPLETE]")


# =====================================================================
# DONE
# =====================================================================
print("\n" + "=" * 60)
print("  WEEKEND PIPELINE COMPLETE")
print("=" * 60)
print("  Next steps:")
print("  1. Clean CSV duplicates")
print("  2. Run 18_master_plotter.py")
print("  3. Compare new results against baseline in thesis")
print("=" * 60)
