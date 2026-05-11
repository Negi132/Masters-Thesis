import os
import sys
import json
import time
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to allow imports from ML_Pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import data_loader
from ML_Pipeline import model_trainer

BASE_EXPERIMENTS_MAP = {
    "Exp1_Weather_Only": ["Weather", "Time"],
    "Exp2_Weather_WeatherLags_Only": ["Weather", "WeatherLags", "Time"],
    "Exp3_Weather_Prices": ["Weather", "Prices", "Time"],
    "Exp4_Weather_WeatherLags_Prices": ["Weather", "WeatherLags", "Prices", "Time"],
    "Exp5_Weather_Grid": ["Weather", "Grid", "GridExchange", "Time"],
    "Exp6_Weather_WeatherLags_Grid": ["Weather", "WeatherLags", "Grid", "GridExchange", "Time"],
    "Exp7_Weather_Grid_Prices": ["Weather", "Grid", "GridExchange", "Prices", "Time"],
    "Exp8_Weather_WeatherLags_Grid_Prices": ["Weather", "WeatherLags", "Grid", "GridExchange", "Prices", "Time"],
    "Exp9_Weather_Grid_Gridlags": ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"],
    "Exp10_Weather_WeatherLags_Grid_Gridlags": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"],
    "Exp11_Weather_Grid_Gridlags_Prices": ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"],
    "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"],
    "Exp13_Total_Information": ["All_Features"]
}

def identify_target_tasks(target_type="Price"):
    print(f"\nScanning CSV for {target_type} architecture benchmarks...")
    try:
        df = pd.read_csv("experiment_results.csv", sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []
        
    df = df[df['Status'] == 'SUCCESS'].copy()
    df['Target_Type'] = df['Target'].astype(str).apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)
    
    def clean_exp_name(name):
        parts = str(name).split('_')
        if len(parts) >= 3 and (parts[-1] in ['Price', 'Delta'] and parts[-2].endswith('h')):
            return "_".join(parts[:-2])
        return name
        
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    df = df[df['Target_Type'] == target_type]
    
    tree_models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
    nn_models = ["LSTM", "GRU", "Transformer", "AutoGluon"]
    
    tasks = []
    
    # 1. Trees (Ranked by mean MAE across all horizons)
    for m in tree_models:
        m_df = df[df['Model'] == m]
        if m_df.empty: continue
        grouped = m_df.groupby('Base_Experiment')['MAE'].mean().reset_index()
        best = grouped.loc[grouped['MAE'].idxmin(), 'Base_Experiment']
        worst = grouped.loc[grouped['MAE'].idxmax(), 'Base_Experiment']
        grouped['Diff'] = abs(grouped['MAE'] - grouped['MAE'].mean())
        mean_exp = grouped.loc[grouped['Diff'].idxmin(), 'Base_Experiment']
        
        tasks.extend([
            {"model": m, "exp": best, "label": "Best"},
            {"model": m, "exp": mean_exp, "label": "Mean"},
            {"model": m, "exp": worst, "label": "Worst"}
        ])
        
    # 2. NNs (Ranked by MAE at the 24h baseline)
    for m in nn_models:
        m_df = df[(df['Model'] == m) & (df['Horizon'] == 24)]
        if m_df.empty: continue
        grouped = m_df.groupby('Base_Experiment')['MAE'].mean().reset_index()
        best = grouped.loc[grouped['MAE'].idxmin(), 'Base_Experiment']
        worst = grouped.loc[grouped['MAE'].idxmax(), 'Base_Experiment']
        grouped['Diff'] = abs(grouped['MAE'] - grouped['MAE'].mean())
        mean_exp = grouped.loc[grouped['Diff'].idxmin(), 'Base_Experiment']
        
        tasks.extend([
            {"model": m, "exp": best, "label": "Best"},
            {"model": m, "exp": mean_exp, "label": "Mean"},
            {"model": m, "exp": worst, "label": "Worst"}
        ])

    # Deduplicate in case mean equals best or worst
    unique_tasks = []
    seen = set()
    for t in tasks:
        sig = f"{t['model']}_{t['exp']}"
        if sig not in seen:
            seen.add(sig)
            unique_tasks.append(t)
            
    return unique_tasks


def execute_pruning_engine(target_type="Price"):
    tasks = identify_target_tasks(target_type)
    if not tasks: return
    
    print(f"\n==================================================")
    print(f" INITIALIZING PRUNING ENGINE FOR: {target_type}")
    print(f"==================================================")
    
    pruned_feature_dict = {}
    output_dir = f"Plots_Feature_Importance_{target_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    # We standardize the permutation test to the 24h horizon
    config.REGION = "DK1"
    config.HORIZON = 24
    config.TARGET_COL = f"TARGET_{target_type}_24h"
    
    for idx, task in enumerate(tasks, 1):
        model_name = task['model']
        base_exp = task['exp']
        label = task['label']
        
        print(f"\n[{idx}/{len(tasks)}] Evaluating {model_name} on {base_exp} ({label} Performer)")
        
        # 1. Load Data
        config.ACTIVE_GROUPS = BASE_EXPERIMENTS_MAP.get(base_exp, ["All_Features"])
        raw_df = data_loader.load_master_data()
        df = data_loader.get_filtered_features(raw_df, active_groups=config.ACTIVE_GROUPS, model_name=model_name)
        
        train_end = int(len(df) * 0.8)
        X_train_raw, y_train, X_test_raw, y_test = data_loader.get_train_test_split(df, 0, train_end, len(df))
        
        # 2. Configure Dynamic Model
        config.RUN_XGBOOST = (model_name == "XGBoost")
        config.RUN_LIGHTGBM = (model_name == "LightGBM")
        config.RUN_CATBOOST = (model_name == "CatBoost")
        config.RUN_RANDOM_FOREST = (model_name == "RandomForest")
        config.RUN_LSTM = (model_name == "LSTM")
        config.RUN_GRU = (model_name == "GRU")
        config.RUN_TRANSFORMER = (model_name == "Transformer")
        config.RUN_AUTOGLUON = (model_name == "AutoGluon")
        
        models_dict = model_trainer.get_models()
        if model_name not in models_dict:
            print(f"  [ERROR] Could not load {model_name}.")
            continue
            
        model = models_dict[model_name]
        
        # 3. Train Baseline
        scaler = None
        if model_name in ["LSTM", "GRU", "Transformer"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = scaler.transform(X_test_raw)
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            
            model.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=0)
            baseline_preds = model.predict(X_test)
            
        elif model_name == "AutoGluon":
            # Clear previous AutoGluon model directory to prevent crash
            if os.path.exists("AutogluonModels"):
                shutil.rmtree("AutogluonModels")
                
            train_data = X_train_raw.copy()
            train_data[config.TARGET_COL] = y_train.values
            model.fit(train_data, label=config.TARGET_COL)
            baseline_preds = model.predict(X_test_raw)
            
        else: # Trees
            fit_kwargs = {}
            if model_name in ["XGBoost", "CatBoost"]:
                fit_kwargs['eval_set'] = [(X_train_raw, y_train), (X_test_raw, y_test)]
                fit_kwargs['verbose'] = False
            elif model_name == "LightGBM":
                fit_kwargs['eval_set'] = [(X_train_raw, y_train), (X_test_raw, y_test)]
                
            model.fit(X_train_raw, y_train, **fit_kwargs)
            baseline_preds = model.predict(X_test_raw)

        baseline_mae = mean_absolute_error(y_test, baseline_preds)
        print(f"  -> Baseline MAE: {baseline_mae:.4f}")
        
        # 4. Custom Universal Permutation Loop
        feature_importances = {}
        features = list(X_test_raw.columns)
        n_repeats = 3
        
        for col in features:
            diffs = []
            for _ in range(n_repeats):
                X_test_shuffled = X_test_raw.copy()
                X_test_shuffled[col] = np.random.permutation(X_test_shuffled[col].values)
                
                if model_name in ["LSTM", "GRU", "Transformer"]:
                    X_test_proc = scaler.transform(X_test_shuffled)
                    X_test_proc = X_test_proc.reshape((X_test_proc.shape[0], 1, X_test_proc.shape[1]))
                    preds = model.predict(X_test_proc)
                elif model_name == "AutoGluon":
                    preds = model.predict(X_test_shuffled)
                else:
                    preds = model.predict(X_test_shuffled)
                    
                perm_mae = mean_absolute_error(y_test, preds)
                diffs.append(perm_mae - baseline_mae)
                
            feature_importances[col] = np.mean(diffs)
            
        # 5. Determine Pruned Dataset
        sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        
        # THRESHOLD: Feature must add at least 0.01 Euro of error when shuffled to be considered "important"
        threshold = 0.01 
        kept_features = [f[0] for f in sorted_features if f[1] > threshold]
        
        # Failsafe: if the architecture is terrible and nothing crosses the threshold, keep top 10
        if len(kept_features) < 5:
            kept_features = [f[0] for f in sorted_features[:10]]
            
        pruned_feature_dict[f"{model_name}_{base_exp}"] = kept_features
        print(f"  -> PRUNED: Kept {len(kept_features)} out of {len(features)} features.")
        
        # 6. Plotting
        top_n = min(20, len(sorted_features))
        plot_features = [f[0] for f in sorted_features[:top_n]]
        plot_impacts = [f[1] for f in sorted_features[:top_n]]
        
        plt.figure(figsize=(12, 8))
        colors = ['steelblue' if val > threshold else 'lightcoral' for val in plot_impacts]
        
        plt.barh(plot_features[::-1], plot_impacts[::-1], color=colors[::-1])
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Pruning Threshold ({threshold})')
        plt.xlabel("Increase in MAE when feature is shuffled (Higher = More Important)")
        plt.title(f"Permutation Importance: {model_name} on {base_exp} ({label})\nTarget: {target_type} | 24h Horizon")
        plt.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"Permutation_{model_name}_{label}_{base_exp}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    # 7. Save the JSON Dictionary
    json_path = f"pruned_features_{target_type}.json"
    with open(json_path, 'w') as f:
        json.dump(pruned_feature_dict, f, indent=4)
        
    print(f"\n[COMPLETE] Successfully exported highly optimized feature sets to {json_path}!")

if __name__ == "__main__":
    execute_pruning_engine(target_type="Price")
    execute_pruning_engine(target_type="Delta")