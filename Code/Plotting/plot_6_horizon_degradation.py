"""
PLOT 6: HORIZON DEGRADATION
============================
Shows how MAE increases with forecast horizon.
Separate subplots for tree models (top) and neural networks (bottom).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import load_csv, get_ylimit_for_plot, apply_cap_annotation

OUTPUT_DIR = "Plot_6_Horizon_Degradation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_horizon_degradation(csv_file, ylimits, separate_by_target, metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return
    
    df = df[df['Version'] == 'Baseline'].copy()
    tree_models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
    nn_models = ["LSTM", "GRU", "Transformer", "AutoGluon"]
    
    for target_type in ['Price', 'Delta']:
        df_t = df[df['Target_Type'] == target_type]
        horizons = sorted(df_t['Horizon'].unique())
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Tree models subplot
        for model in tree_models:
            model_data = df_t[df_t['Model'] == model]
            best_vals = [model_data[model_data['Horizon']==h][metric].min() 
                        if not model_data[model_data['Horizon']==h].empty else np.nan 
                        for h in horizons]
            ax1.plot(horizons, best_vals, marker='o', label=model, linewidth=2)
        
        ax1.set_title(f"Tree Models: {metric} Degradation vs Horizon\nTarget: {target_type}", 
                     fontsize=14)
        ax1.set_xlabel("Forecast Horizon (hours)", fontsize=12)
        ax1.set_ylabel(f"Best {metric} (Lower is better)", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # NN models subplot
        for model in nn_models:
            model_data = df_t[df_t['Model'] == model]
            best_vals = [model_data[model_data['Horizon']==h][metric].min() 
                        if not model_data[model_data['Horizon']==h].empty else np.nan 
                        for h in horizons]
            ax2.plot(horizons, best_vals, marker='s', label=model, linewidth=2)
        
        ax2.set_title(f"Neural Networks: {metric} Degradation vs Horizon\nTarget: {target_type}", 
                     fontsize=14)
        ax2.set_xlabel("Forecast Horizon (hours)", fontsize=12)
        ax2.set_ylabel(f"Best {metric} (Lower is better)", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Apply unified y-limits
        if ylimits:
            cap = get_ylimit_for_plot(ylimits, 'horizon_model', target_type, separate_by_target)
            if cap:
                ax1.set_ylim(top=cap)
                ax2.set_ylim(top=cap)
                apply_cap_annotation(ax1, cap, metric)
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"Horizon_Degradation_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


def generate_all_plots(csv_file, ylimits, separate_by_target):
    plot_horizon_degradation(csv_file, ylimits, separate_by_target)


if __name__ == "__main__":
    from plot_utils import compute_unified_ylimits
    csv_file = "../experiment_results_clean.csv"
    df = load_csv(csv_file)
    df_baseline = df[df['Version'] == 'Baseline'].copy()
    ylimits = compute_unified_ylimits(df_baseline, 2.5, 3.0, True)
    generate_all_plots(csv_file, ylimits, True)
