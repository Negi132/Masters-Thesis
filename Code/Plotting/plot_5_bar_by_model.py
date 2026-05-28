"""
PLOT 5: BAR CHARTS BY MODEL
============================
Shows Best/Mean/Worst MAE per horizon for each model.
One plot per model, per target, per version.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import load_csv, get_ylimit_for_plot, apply_cap_annotation

OUTPUT_DIR = "Plot_5_Bar_By_Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_bar_by_model(csv_file, model_name, target_type, version_filter,
                     ylimits, separate_by_target, metric="MAE"):
    """
    Generate bar chart showing Best/Mean/Worst performance across horizons for one model.
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    df = df[(df['Version'] == version_filter) & 
            (df['Target_Type'] == target_type) &
            (df['Model'] == model_name)].copy()
    
    if df.empty:
        print(f"  [SKIP] No data for {model_name} {target_type} {version_filter}")
        return
    
    horizons = sorted(df['Horizon'].unique())
    x_positions = np.arange(len(horizons))
    
    bar_colors = {'Best': '#2ecc71', 'Mean': '#f39c12', 'Worst': '#e74c3c'}
    bar_width = 0.25
    x_offsets = {'Best': -bar_width, 'Mean': 0, 'Worst': bar_width}
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for rank_label in ['Best', 'Mean', 'Worst']:
        values = []
        for horizon in horizons:
            horizon_data = df[df['Horizon'] == horizon][metric].dropna()
            if horizon_data.empty:
                values.append(np.nan)
            elif rank_label == 'Best':
                values.append(horizon_data.min())
            elif rank_label == 'Worst':
                values.append(horizon_data.max())
            else:
                values.append(horizon_data.mean())
        
        bars = ax.bar(
            x_positions + x_offsets[rank_label],
            values, width=bar_width, label=rank_label,
            color=bar_colors[rank_label], alpha=0.85, 
            edgecolor='white', linewidth=0.5
        )
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    cap = None
    if ylimits:
        cap = get_ylimit_for_plot(ylimits, 'horizon_model', target_type, separate_by_target)
        if cap:
            ax.set_ylim(top=cap)
            apply_cap_annotation(ax, cap, metric)
            for bar in ax.patches:
                if bar.get_height() >= cap * 0.98:
                    ax.text(bar.get_x() + bar.get_width()/2, cap * 0.97, 
                           f'{bar.get_height():.0f}', ha='center', va='top',
                           fontsize=7, color='white', fontweight='bold')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{h}h" for h in horizons], fontsize=10)
    ax.set_xlabel("Forecast Horizon", fontsize=12)
    ax.set_ylabel(f"{metric} (Lower is better)", fontsize=12)
    ax.set_title(f"Best / Mean / Worst {metric} across Horizons\n"
                f"Model: {model_name} | Target: {target_type} | Version: {version_filter}", 
                fontsize=14)
    ax.legend(title="Feature Set Performance", fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Tight layout.*')
        plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 
                            f"Bar_ByModel_{model_name}_{target_type}_{version_filter}_{metric}.png")
    plt.savefig(save_path, dpi=300)
    print(f"    ✓ {os.path.basename(save_path)}")
    plt.close()


def generate_all_plots(csv_file, version_filter, ylimits, separate_by_target):
    df = load_csv(csv_file)
    if df is None:
        return
    
    df_version = df[df['Version'] == version_filter].copy()
    models = sorted(df_version['Model'].unique())
    
    print(f"  Generating plots for version: {version_filter}")
    
    for target_type in ['Price', 'Delta']:
        for model_name in models:
            plot_bar_by_model(csv_file, model_name, target_type, version_filter,
                            ylimits, separate_by_target)


if __name__ == "__main__":
    from plot_utils import compute_unified_ylimits
    csv_file = "../experiment_results_clean.csv"
    df = load_csv(csv_file)
    df_baseline = df[df['Version'] == 'Baseline'].copy()
    ylimits = compute_unified_ylimits(df_baseline, 2.5, 3.0, True)
    generate_all_plots(csv_file, "Baseline", ylimits, True)
