"""
Diagnose supplementary plot issues
===================================
1. Check why Naive Persistence bars are missing
2. Verify Optuna data is correct
3. Check Midas data anomalies
"""
import pandas as pd
import numpy as np

csv_path = "../ML_Pipeline/experiment_results_clean.csv"
df = pd.read_csv(csv_path, sep=None, engine='python')

print("=" * 80)
print("1. NAIVE BASELINE INVESTIGATION")
print("=" * 80)

naive = df[df['Model'] == 'Naive_Persistence'].copy()
naive['Horizon'] = naive['Target'].str.extract(r'(\d+)h').astype(int)
naive['Target_Type'] = naive['Target'].str.split('_').str[1]

print(f"\nTotal Naive Persistence rows: {len(naive)}")
print(f"\nNaive MAE values:")
print(f"{'Target':<10} {'Horizon':<10} {'MAE':>15}")
print("-" * 40)
for target in ['Price', 'Delta']:
    target_data = naive[naive['Target_Type'] == target].sort_values('Horizon')
    for _, row in target_data.iterrows():
        print(f"{target:<10} {row['Horizon']:>3}h       {row['MAE']:>15.2f}")

print("\n" + "=" * 80)
print("2. BEST ML MODEL COMPARISON")
print("=" * 80)

# Get best ML models for comparison
ml_df = df[
    (df['Model'] != 'Naive_Persistence') &
    (~df['Experiment'].str.contains('Optuna|Midas|MAELoss|DK2', case=False, na=False))
].copy()
ml_df['Horizon'] = ml_df['Target'].str.extract(r'(\d+)h').astype(int)
ml_df['Target_Type'] = ml_df['Target'].str.split('_').str[1]

print(f"\nComparison (Naive vs Best ML):")
print(f"{'Target':<10} {'Horizon':<10} {'Naive MAE':>12} {'Best ML MAE':>15} {'Naive/ML Ratio':>15}")
print("-" * 65)
for target in ['Price', 'Delta']:
    horizons = [0, 24, 48, 72, 96, 120, 144, 168] if target == 'Price' else [24, 48, 72, 96, 120, 144, 168]
    for h in horizons:
        naive_mae = naive[(naive['Target_Type']==target) & (naive['Horizon']==h)]['MAE']
        ml_mae = ml_df[(ml_df['Target_Type']==target) & (ml_df['Horizon']==h)]['MAE']
        
        if len(naive_mae) > 0 and len(ml_mae) > 0:
            n = naive_mae.min()
            m = ml_mae.min()
            ratio = n / m if m > 0 else float('inf')
            print(f"{target:<10} {h:>3}h       {n:>12.2f} {m:>15.2f} {ratio:>15.2f}x")

print("\n" + "=" * 80)
print("3. OPTUNA INVESTIGATION")
print("=" * 80)

optuna_df = df[df['Experiment'].str.contains('Optuna', case=False, na=False)].copy()
optuna_df['Target_Type'] = optuna_df['Target'].str.split('_').str[1]
optuna_df['Horizon'] = optuna_df['Target'].str.extract(r'(\d+)h').astype(int)

print(f"\nOptuna rows: {len(optuna_df)}")
print(f"\nOptuna results vs Default baseline:")
print(f"{'Target':<10} {'Model':<15} {'Default MAE':>15} {'Tuned MAE':>15} {'Change %':>12}")
print("-" * 70)

for target in ['Price', 'Delta']:
    for model in ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest']:
        optuna_mae = optuna_df[
            (optuna_df['Model']==model) & 
            (optuna_df['Target_Type']==target) & 
            (optuna_df['Horizon']==24)
        ]['MAE']
        
        default_mae = ml_df[
            (ml_df['Model']==model) & 
            (ml_df['Target_Type']==target) & 
            (ml_df['Horizon']==24)
        ]['MAE'].mean()
        
        if len(optuna_mae) > 0:
            tuned = optuna_mae.mean()
            change = ((tuned - default_mae) / default_mae) * 100
            print(f"{target:<10} {model:<15} {default_mae:>15.2f} {tuned:>15.2f} {change:>11.1f}%")

print("\n" + "=" * 80)
print("4. MIDAS WEATHER INVESTIGATION")
print("=" * 80)

midas_df = df[df['Experiment'].str.contains('Midas', case=False, na=False)].copy()
midas_df['Target_Type'] = midas_df['Target'].str.split('_').str[1]
midas_df['Horizon'] = midas_df['Target'].str.extract(r'(\d+)h').astype(int)

print(f"\nMidas rows: {len(midas_df)}")
print(f"\nMidas vs DMI comparison:")
print(f"{'Target':<10} {'Model':<15} {'DMI MAE':>12} {'Midas MAE':>12} {'Change %':>12}")
print("-" * 65)

for target in ['Price', 'Delta']:
    for model in ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest', 'LSTM', 'GRU', 'Transformer', 'AutoGluon']:
        midas_mae = midas_df[
            (midas_df['Model']==model) & 
            (midas_df['Target_Type']==target) & 
            (midas_df['Horizon']==24)
        ]['MAE']
        
        dmi_mae = ml_df[
            (ml_df['Model']==model) & 
            (ml_df['Target_Type']==target) & 
            (ml_df['Horizon']==24)
        ]['MAE'].mean()
        
        if len(midas_mae) > 0:
            m = midas_mae.mean()
            change = ((m - dmi_mae) / dmi_mae) * 100
            print(f"{target:<10} {model:<15} {dmi_mae:>12.2f} {m:>12.2f} {change:>11.1f}%")
