"""
Optuna objective with walk-forward validation
==============================================
The previous Optuna implementation optimised against a single 30-day
validation slice at the very end of the dataset, which is not what
the rest of the pipeline measures. Hyperparameters found that way
overfit one specific market period.

This module provides a walk-forward objective: each trial trains and
evaluates the candidate hyperparameters on N spread-out folds and
returns the mean MAE. This matches what the final reported MAE
actually measures.

Three folds is the default. Three is enough to:
  - See different seasons / market conditions per trial
  - Detect overfitting to any single window
  - Keep trial cost bounded (3x a single fit, not 30x)

Folds are chosen evenly spaced across the dataset.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor


def build_model(trial, model_type):
    """Returns a configured model for the trial. Search spaces match the
    original weekend script."""
    if model_type == "XGBoost":
        params = {
            'max_depth':       trial.suggest_int('max_depth', 3, 10),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':    trial.suggest_int('n_estimators', 50, 500),
            'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_jobs': -1,
        }
        return XGBRegressor(**params)
    if model_type == "LightGBM":
        params = {
            'num_leaves':        trial.suggest_int('num_leaves', 20, 150),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators':      trial.suggest_int('n_estimators', 50, 500),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'n_jobs': -1, 'verbose': -1,
        }
        return LGBMRegressor(**params)
    if model_type == "CatBoost":
        params = {
            'depth':         trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations':    trial.suggest_int('iterations', 50, 500),
            'verbose': 0,
        }
        return CatBoostRegressor(**params)
    if model_type == "RandomForest":
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 50, 300),
            'max_depth':         trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'n_jobs': -1,
        }
        return RandomForestRegressor(**params)
    raise ValueError(f"Unknown model_type: {model_type}")


def build_walkforward_folds(df, target_col, n_folds=3,
                             train_hours_per_fold=730 * 24,
                             test_hours_per_fold=30 * 24):
    """
    Returns a list of (X_train, y_train, X_val, y_val) tuples spread
    evenly across the dataset. By default each fold trains on ~2 years
    and validates on 30 days, which mirrors the production walk-forward
    cadence.

    Default n_folds=3 chosen to keep cost bounded while still sampling
    multiple market periods. Increase for a more robust objective at
    higher cost.

    Folds are positioned so the LAST fold ends near the end of the
    dataset, and earlier folds are spread evenly backwards. This
    guarantees enough data for the initial train window of each fold.
    """
    total = len(df)
    fold_size = train_hours_per_fold + test_hours_per_fold

    if total < fold_size:
        raise ValueError(
            f"Dataset too short for walk-forward folds. "
            f"Need >= {fold_size} hours, have {total}.")

    # Spread fold END points evenly. Last fold ends at total.
    # First fold ends at fold_size (earliest possible).
    if n_folds == 1:
        end_points = [total]
    else:
        end_points = np.linspace(fold_size, total, n_folds).astype(int).tolist()

    folds = []
    drop_cols = [target_col]
    if 'HourUTC' in df.columns:
        drop_cols.append('HourUTC')

    for end in end_points:
        train_end = end - test_hours_per_fold
        train_start = train_end - train_hours_per_fold
        train_slice = df.iloc[train_start:train_end]
        val_slice = df.iloc[train_end:end]

        X_tr = train_slice.drop(columns=drop_cols, errors='ignore')
        y_tr = train_slice[target_col]
        X_val = val_slice.drop(columns=drop_cols, errors='ignore')
        y_val = val_slice[target_col]
        folds.append((X_tr, y_tr, X_val, y_val))
    return folds


def walkforward_objective(trial, model_type, folds):
    """
    Trains a fresh model on each fold and returns the mean validation MAE.
    Each fold gets a NEW model instance via build_model() so internal
    state from one fold cannot leak into the next.
    """
    maes = []
    for X_tr, y_tr, X_val, y_val in folds:
        model = build_model(trial, model_type)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))
    return float(np.mean(maes))
