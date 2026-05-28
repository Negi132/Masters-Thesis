"""
PLOT 7: MODEL COMPARISON
========================
Head-to-head bar chart showing best MAE per model at each horizon.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import load_csv, get_ylimit_for_plot, apply_cap_annotation

OUTPUT_DIR = "Plot_7_Model_Comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_model_comparison(csv_file, ylimits, separate_by_target, metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return
    
    df = df[df['Version'] == 'Baseline'].copy()
    
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
        
        x_positions = np.arange(len(horizons))
        bar_width = 0.08
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        for i, model in enumerate(models):
            offset = (i - len(models)/2 + 0.5) * bar_width
            values = []
            for horizon in horizons:
                model_data = df_t[(df_t['Model']==model) & (df_t['Horizon']==horizon)][metric]
                values.append(model_data.min() if not model_data.empty else np.nan)
            
            bars = ax.bar(x_positions + offset, values, width=bar_width,
                         label=model, color=model_colors.get(model, 'grey'), alpha=0.85)
            
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.0f}', ha='center', va='bottom', fontsize=5.5, rotation=90)
        
        if ylimits:
            cap = get_ylimit_for_plot(ylimits, 'horizon_model', target_type, separate_by_target)
            if cap:
                ax.set_ylim(top=cap)
                apply_cap_annotation(ax, cap, metric)
                for bar in ax.patches:
                    if bar.get_height() >= cap * 0.98:
                        ax.text(bar.get_x() + bar.get_width()/2, cap * 0.97,
                               f'{bar.get_height():.0f}', ha='center', va='top',
                               fontsize=5.5, color='white', fontweight='bold', rotation=90)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{h}h" for h in horizons])
        ax.set_xlabel("Forecast Horizon", fontsize=12)
        ax.set_ylabel(f"Best {metric} per Model (Lower is better)", fontsize=12)
        ax.set_title(f"Model Comparison: Best {metric} at Each Horizon\nTarget: {target_type}",
                    fontsize=14)
        ax.legend(ncol=4, fontsize=9, loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Tight layout.*')
            plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"Model_Comparison_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


def generate_all_plots(csv_file, ylimits, separate_by_target):
    plot_model_comparison(csv_file, ylimits, separate_by_target)


if __name__ == "__main__":
    from plot_utils import compute_unified_ylimits
    csv_file = "../experiment_results_clean.csv"
    df = load_csv(csv_file)
    df_baseline = df[df['Version'] == 'Baseline'].copy()
    ylimits = compute_unified_ylimits(df_baseline, 2.5, 3.0, True)
    generate_all_plots(csv_file, ylimits, True)
