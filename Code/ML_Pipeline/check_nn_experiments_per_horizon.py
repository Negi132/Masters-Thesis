"""
Complete NN experiment listing per horizon
==========================================
Shows EVERY experiment that was run for LSTM, GRU, Transformer
at each horizon and target. No truncation.

This output is the definitive list of experiments to replicate
with MAE loss for proper comparison.
"""
import pandas as pd
import numpy as np

csv_path = "experiment_results_clean.csv"
df = pd.read_csv(csv_path, sep=None, engine='python')

# Process targets and horizons
df = df[df['Status'] == 'SUCCESS'].copy()
df['Target_Type'] = df['Target'].astype(str).apply(
    lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
df['Horizon'] = df['Target'].astype(str).apply(
    lambda x: int(x.split('_')[2].replace('h','')) if len(x.split('_')) > 2 else -1)

def clean_exp_name(name):
    for tag in ['_0h','_24h','_48h','_72h','_96h','_120h','_144h','_168h']:
        if tag in str(name):
            return str(name).split(tag)[0]
    return str(name)

df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)

# Exclude special experiments
df = df[~df['Experiment'].str.contains(
    'Pruned|FullWeek|Fullweek|Midas|Optuna|MAELoss|DK2|Naive',
    case=False, na=False)]

NN_MODELS = ['LSTM', 'GRU', 'Transformer']

# Track all unique (model, target, horizon, experiment) combinations
all_combinations = []

for nn in NN_MODELS:
    print("=" * 80)
    print(f"  MODEL: {nn}")
    print("=" * 80)
    
    df_nn = df[df['Model'] == nn]
    
    for target in ['Price', 'Delta']:
        df_t = df_nn[df_nn['Target_Type'] == target]
        horizons = sorted(df_t['Horizon'].unique())
        
        if not horizons:
            continue
        
        print(f"\n  Target: {target}")
        print(f"  {'-' * 78}")
        
        for h in horizons:
            df_h = df_t[df_t['Horizon'] == h]
            experiments = sorted(df_h['Base_Experiment'].unique())
            
            print(f"\n  Horizon: {h}h  ({len(experiments)} experiments)")
            for exp in experiments:
                mae = df_h[df_h['Base_Experiment'] == exp]['MAE'].mean()
                print(f"    - {exp:<60} (MAE: {mae:.2f})")
                all_combinations.append({
                    'Model': nn,
                    'Target': target,
                    'Horizon': h,
                    'Experiment': exp,
                    'MAE': mae
                })

# Summary statistics
print("\n" + "=" * 80)
print("  SUMMARY: EXPERIMENTS TO REPLICATE WITH MAE LOSS")
print("=" * 80)

total = len(all_combinations)
print(f"\n  Total experiments across all NN models: {total}")

# Count per model
for nn in NN_MODELS:
    count = sum(1 for c in all_combinations if c['Model'] == nn)
    print(f"    {nn}: {count} experiments")

# Identify the "non-24h" experiments per model+target+horizon (since 24h has all 13)
# These are the "best/mean/worst" estimates from previous runs
print("\n" + "=" * 80)
print("  EXPERIMENTS PER HORIZON (excluding 24h which has all 13)")
print("=" * 80)

non_24h = [c for c in all_combinations if c['Horizon'] != 24]
print(f"\n  Total non-24h experiments: {len(non_24h)}")

# Group by model+target+horizon
print(f"\n  Detailed listing of feature sets used per horizon:")
print(f"  (These are the 'best/mean/worst' estimates from baseline runs)")

for nn in NN_MODELS:
    for target in ['Price', 'Delta']:
        print(f"\n  {nn} | {target}:")
        for h in sorted(set(c['Horizon'] for c in non_24h 
                            if c['Model'] == nn and c['Target'] == target)):
            exps = sorted([c['Experiment'] for c in non_24h 
                          if c['Model'] == nn and c['Target'] == target and c['Horizon'] == h])
            print(f"    {h:>3}h: {', '.join(exps)}")

# Save to CSV for reference
output_df = pd.DataFrame(all_combinations)
output_df.to_csv("nn_experiments_to_replicate.csv", index=False)
print(f"\n  Full list saved to: nn_experiments_to_replicate.csv")
print(f"  Total rows: {len(output_df)}")
