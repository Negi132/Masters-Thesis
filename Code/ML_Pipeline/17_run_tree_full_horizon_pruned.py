import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
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

def identify_tree_experiments():
    print("Analyzing CSV to find optimal Tree Model test sets...")
    
    try:
        df = pd.read_csv("experiment_results.csv", sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None
        
    df = df[df['Status'] == 'SUCCESS'].copy()
    df = df[~df['Experiment'].str.contains('Pruned|FullWeek|Fullweek', case=False, na=False)]
    
    df['Target_Type'] = df['Target'].astype(str).apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)
    
    def clean_exp_name(name):
        parts = str(name).split('_')
        if len(parts) >= 3 and (parts[-1] in ['Price', 'Delta'] and parts[-2].endswith('h')):
            return "_".join(parts[:-2])
        return name
        
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    
    tree_models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
    df_trees = df[df['Model'].isin(tree_models)].copy()
    
    tasks_price = set()
    tasks_delta = set()
    
    # Identify Best/Mean/Worst for each model based on overall MAE average
    for target in ["Price", "Delta"]:
        df_t = df_trees[df_trees['Target_Type'] == target]
        for m in tree_models:
            m_df = df_t[df_t['Model'] == m]
            if m_df.empty: continue
            
            grouped = m_df.groupby('Base_Experiment')['MAE'].mean().reset_index()
            best = grouped.loc[grouped['MAE'].idxmin(), 'Base_Experiment']
            worst = grouped.loc[grouped['MAE'].idxmax(), 'Base_Experiment']
            
            grouped['Diff'] = abs(grouped['MAE'] - grouped['MAE'].mean())
            mean_exp = grouped.loc[grouped['Diff'].idxmin(), 'Base_Experiment']
            
            if target == "Price":
                tasks_price.update([best, mean_exp, worst])
            else:
                tasks_delta.update([best, mean_exp, worst])
                
    return list(tasks_price), list(tasks_delta)

def main():
    print("==================================================")
    print("      PRUNED TREE FULL HORIZON SUITE              ")
    print("==================================================")
    
    price_exps, delta_exps = identify_tree_experiments()
    if not price_exps:
        return
        
    experiment_queue = []
    
    # Queue up Price experiments (0h to 168h)
    for horizon in range(0, 169, 24):
        for base in price_exps:
            experiment_queue.append({
                "name": f"{base}_{horizon}h_Price_Pruned",
                "groups": BASE_EXPERIMENTS_MAP[base],
                "horizon": horizon,
                "target": "Price"
            })
            
    # Queue up Delta experiments (24h to 168h)
    for horizon in range(24, 169, 24):
        for base in delta_exps:
            experiment_queue.append({
                "name": f"{base}_{horizon}h_Delta_Pruned",
                "groups": BASE_EXPERIMENTS_MAP[base],
                "horizon": horizon,
                "target": "Delta"
            })

    print(f"Queue generated! Executing {len(experiment_queue)} highly targeted, PRUNED experiments.")
    print("="*50)
    
    total_start_time = time.time()

    for i, exp in enumerate(experiment_queue, 1):
        print(f"\n[{i}/{len(experiment_queue)}] INITIALIZING: {exp['name']}")
        
        config.EXPERIMENT_NAME = exp["name"]
        config.ACTIVE_GROUPS = exp["groups"]
        config.REGION = "DK1"
        config.HORIZON = exp["horizon"]
        config.TARGET_COL = f"TARGET_{exp['target']}_{exp['horizon']}h"
        
        try:
            model_trainer.run_walk_forward_pipeline()
            print(f"  -> [SUCCESS] {exp['name']} completed.")
        except Exception as e:
            print(f"  -> [ERROR] {exp['name']} failed: {e}")

    total_time = (time.time() - total_start_time) / 3600
    print("\n" + "="*50)
    print(f"PRUNED FULL HORIZON SUITE FINISHED in {total_time:.2f} hours.")
    print("="*50)

if __name__ == "__main__":
    main()