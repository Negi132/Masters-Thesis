"""
PLOT 9: PRUNING GAIN
====================
Shows percentage MAE improvement from baseline to pruned feature sets.
Positive values = pruning helped, negative = pruning hurt.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import load_csv

OUTPUT_DIR = "Plot_9_Pruning_Gain"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_pruning_gain(csv_file, metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return
    
    model_colors = {
        "CatBoost": "#e74c3c", "LightGBM": "#e67e22",
        "XGBoost": "#f1c40f", "RandomForest": "#2ecc71",
        "LSTM": "#3498db", "GRU": "#9b59b6",
        "Transformer": "#1abc9c", "AutoGluon": "#e91e63"
    }
    
    for target_type in ['Price', 'Delta']:
        df_t = df[df['Target_Type'] == target_type]
        horizons = sorted(df_t['Horizon'].unique())
        models = sorted(df_t['Model'].unique())
        
        bar_width = 0.1
        x_positions = np.arange(len(horizons))
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Add vertical separators between horizons
        for i in range(1, len(horizons)):
            ax.axvline(x=i - 0.5, color='gray', linestyle=':', alpha=0.6)
        
        any_data = False
        for j, model_name in enumerate(models):
            gains = []
            for horizon in horizons:
                base_data = df_t[(df_t['Version']=='Baseline') & 
                                (df_t['Model']==model_name) & 
                                (df_t['Horizon']==horizon)][metric].dropna()
                pruned_data = df_t[(df_t['Version']=='Pruned') & 
                                  (df_t['Model']==model_name) & 
                                  (df_t['Horizon']==horizon)][metric].dropna()
                
                if base_data.empty or pruned_data.empty:
                    gains.append(np.nan)
                else:
                    base_best = base_data.min()
                    pruned_best = pruned_data.min()
                    pct_gain = ((base_best - pruned_best) / base_best) * 100
                    gains.append(pct_gain)
            
            if all(np.isnan(g) for g in gains):
                continue
            
            any_data = True
            offset = (j - len(models)/2 + 0.5) * bar_width
            ax.bar(x_positions + offset, gains, width=bar_width, label=model_name,
                  color=model_colors.get(model_name, 'grey'), alpha=0.85,
                  edgecolor='white', linewidth=0.5)
        
        if not any_data:
            print(f"  [SKIP] No pruning data for {target_type}")
            plt.close()
            continue
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{h}h" for h in horizons])
        ax.set_xlabel("Forecast Horizon", fontsize=12)
        ax.set_ylabel("% MAE Improvement (Higher is better)", fontsize=12)
        ax.set_title(f"Pruning Gain: Baseline vs Pruned\nTarget: {target_type}", fontsize=14)
        ax.legend(ncol=4, fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Symmetric y-axis
        ylim = ax.get_ylim()
        max_abs = max(abs(ylim[0]), abs(ylim[1]))
        ax.set_ylim(-max_abs, max_abs)
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f"Pruning_Gain_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


def generate_all_plots(csv_file):
    plot_pruning_gain(csv_file)


if __name__ == "__main__":
    generate_all_plots("../experiment_results_clean.csv")
