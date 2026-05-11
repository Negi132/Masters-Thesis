import time
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ML_Pipeline import config
from ML_Pipeline import model_trainer

# Hardcoded mapping of feature sets so we don't have to parse strings
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

def identify_target_experiments():
    print("Analyzing CSV to find optimal Neural Network test sets...")
    
    try:
        df = pd.read_csv("experiment_results.csv", sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None
        
    df = df[df['Status'] == 'SUCCESS'].copy()

    # Remove polluted runs
    df = df[~df['Experiment'].str.contains('FullWeek|Fullweek|Pruned', case=False, na=False)]
    
    # Extract Target and Horizon
    df['Target_Type'] = df['Target'].astype(str).apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)
    
    def clean_exp_name(name):
        base_exp = str(name)
        # Bulletproof slice: finds the horizon tag and chops off everything after it
        for split_str in ['_0h', '_24h', '_48h', '_72h', '_96h', '_120h', '_144h', '_168h']:
            if split_str in base_exp:
                base_exp = base_exp.split(split_str)[0]
                break
        return base_exp
        
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    
    # Filter down to the 24h horizon where we ran ALL 13 experiments for NNs
    nn_models = ['LSTM', 'GRU', 'Transformer', 'AutoGluon']
    df_nn = df[(df['Model'].isin(nn_models)) & (df['Horizon'] == 24)].copy()
    
    def get_top_3(target_type, metric):
        temp = df_nn[df_nn['Target_Type'] == target_type]
        if temp.empty:
            print(f"  [WARNING] No 24h NN data found for {target_type}. Falling back to Exp13.")
            return ["Exp13_Total_Information"] * 3
            
        # Group by Base_Experiment and get the average metric across all NNs
        grouped = temp.groupby('Base_Experiment')[metric].mean().reset_index()
        
        best_exp = grouped.loc[grouped[metric].idxmin(), 'Base_Experiment']
        worst_exp = grouped.loc[grouped[metric].idxmax(), 'Base_Experiment']
        
        overall_mean = grouped[metric].mean()
        grouped['Diff'] = abs(grouped[metric] - overall_mean)
        mean_exp = grouped.loc[grouped['Diff'].idxmin(), 'Base_Experiment']
        
        return best_exp, mean_exp, worst_exp

    # Price uses WMAPE, Delta uses MAE (to avoid percentage explosion)
    p_best, p_mean, p_worst = get_top_3("Price", "WMAPE")
    d_best, d_mean, d_worst = get_top_3("Delta", "MAE")
    
    print("\n--- IDENTIFIED FEATURE SETS ---")
    print(f"PRICE: Best=[{p_best}], Mean=[{p_mean}], Worst=[{p_worst}]")
    print(f"DELTA: Best=[{d_best}], Mean=[{d_mean}], Worst=[{d_worst}]")
    print("-------------------------------\n")
    
    # We use sets to remove duplicates (in case the mean happens to equal the best or worst)
    return list(set([p_best, p_mean, p_worst])), list(set([d_best, d_mean, d_worst]))


def main():
    print("==================================================")
    print("      NN FULL HORIZON SENSITIVITY SUITE           ")
    print("==================================================")
    
    price_exps, delta_exps = identify_target_experiments()
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

    print(f"Queue generated! Executing {len(experiment_queue)} highly targeted experiments.")
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
    print(f"FULL HORIZON SUITE FINISHED in {total_time:.2f} hours.")
    print("="*50)

if __name__ == "__main__":
    main()