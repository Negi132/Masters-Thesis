import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from ML_Pipeline import config
from datetime import datetime

def calculate_metrics(y_true, y_pred, train_time, status="SUCCESS"):
    """Calculates all 7 thesis metrics with safeguards for zero/negative prices."""
    
    if status != "SUCCESS":
        return {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Experiment": config.EXPERIMENT_NAME,
            "Region": config.REGION,
            "Target": config.TARGET_COL,
            "Feature_Mask": str(config.ACTIVE_GROUPS),
            "Status": status,
            "RMSE": np.nan, "MAE": np.nan, "R2": np.nan, 
            "WMAPE": np.nan, "sMAPE": np.nan, "MDA": np.nan, 
            "Train_Time_Sec": 0
        }

    # Ensure inputs are numpy arrays for vector math
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. Standard Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 2. WMAPE (Weighted Mean Absolute Percentage Error)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        wmape = 0.0 # Failsafe for entirely flat actuals
    else:
        wmape = (np.sum(np.abs(y_true - y_pred)) / denominator) * 100
    
    # 3. sMAPE (Symmetric Mean Absolute Percentage Error)
    # Added 1e-8 epsilon to prevent division by zero on 0.00 EUR price hours
    smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    
    # 4. MDA (Mean Directional Accuracy)
    # Did we correctly predict the TREND (Up/Down)?
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    mda = np.mean((np.sign(actual_diff) == np.sign(pred_diff)).astype(int)) * 100

    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Experiment": config.EXPERIMENT_NAME,
        "Region": config.REGION,
        "Target": config.TARGET_COL,
        "Feature_Mask": str(config.ACTIVE_GROUPS),
        "Status": status,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "WMAPE": round(wmape, 4),
        "sMAPE": round(smape, 4),
        "MDA": round(mda, 4),
        "Train_Time_Sec": round(train_time, 2)
    }

def log_experiment(results_dict):
    """Appends results to CSV. Header is created only if file is new."""
    file_exists = os.path.isfile(config.EXPERIMENT_LOG)
    pd.DataFrame([results_dict]).to_csv(
        config.EXPERIMENT_LOG, mode='a', index=False, header=not file_exists
    )
    print(f"  -> Results logged to {config.EXPERIMENT_LOG}")