"""
PLOT 4: BAR CHARTS BY HORIZON
==============================
Shows Best/Mean/Worst MAE per model at each horizon.
One plot per horizon, per target, per version.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import load_csv, get_ylimit_for_plot, apply_cap_annotation

OUTPUT_DIR = "Plot_4_Bar_By_Horizon"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_bar_by_horizon(csv_file, target_type, horizon, version_filter, 
                        ylimits, separate_by_target, metric="MAE"):
    """
    Generate bar chart comparing Best/Mean/Worst performance per model at a specific horizon.
    
    Args:
        csv_file: Path to experiment results CSV
        target_type: 'Price' or 'Delta'
        horizon: Forecasting horizon in hours
        version_filter: 'Baseline' or 'Pruned'
        ylimits: Dict of unified y-axis limits from compute_unified_ylimits()
        separate_by_target: If True, use target-specific y-limits
        metric: Evaluation metric to plot (default: MAE)
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # Filter to specific version and target
    df = df[(df['Version'] == version_filter) & 
            (df['Target_Type'] == target_type) & 
            (df['Horizon'] == horizon)].copy()
    
    if df.empty:
        print(f"  [SKIP] No data for {target_type} {horizon}h {version_filter}")
        return
    
    models = sorted(df['Model'].unique())
    x_positions = np.arange(len(models))
    
    bar_colors = {'Best': '#2ecc71', 'Mean': '#f39c12', 'Worst': '#e74c3c'}
    bar_width = 0.25
    x_offsets = {'Best': -bar_width, 'Mean': 0, 'Worst': bar_width}
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot bars for each rank category
    for rank_label in ['Best', 'Mean', 'Worst']:
        values = []
        for model in models:
            model_data = df[df['Model'] == model][metric].dropna()
            if model_data.empty:
                values.append(np.nan)
                continue
            
            if rank_label == 'Best':
                values.append(model_data.min())
            elif rank_label == 'Worst':
                values.append(model_data.max())
            else:  # Mean
                values.append(model_data.mean())
        
        bars = ax.bar(
            x_positions + x_offsets[rank_label],
            values,
            width=bar_width,
            label=rank_label,
            color=bar_colors[rank_label],
            alpha=0.85,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add value labels above bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=7, rotation=45
                )
    
    # Apply unified y-axis limit
    cap = None
    if ylimits:
        cap = get_ylimit_for_plot(ylimits, 'horizon_model', target_type, separate_by_target)
        if cap:
            ax.set_ylim(top=cap)
            apply_cap_annotation(ax, cap, metric)
            
            # Mark bars that were truncated
            for bar in ax.patches:
                true_val = bar.get_height()
                if true_val >= cap * 0.98:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        cap * 0.97,
                        f'{true_val:.0f}',
                        ha='center', va='top', fontsize=7,
                        color='white', fontweight='bold'
                    )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_xlabel("Model Architecture", fontsize=12)
    ax.set_ylabel(f"{metric} (Lower is better)", fontsize=12)
    ax.set_title(
        f"Best / Mean / Worst {metric} by Model Architecture\n"
        f"Target: {target_type} | Horizon: {horizon}h | Version: {version_filter}",
        fontsize=14
    )
    ax.legend(title="Feature Set Performance", fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Tight layout.*')
        plt.tight_layout()
    
    save_path = os.path.join(
        OUTPUT_DIR,
        f"Bar_ByHorizon_{target_type}_{horizon}h_{version_filter}_{metric}.png"
    )
    plt.savefig(save_path, dpi=300)
    print(f"    ✓ {os.path.basename(save_path)}")
    plt.close()


def generate_all_plots(csv_file, version_filter, ylimits, separate_by_target):
    """
    Generate all bar-by-horizon plots for the specified version.
    
    Args:
        csv_file: Path to experiment results CSV
        version_filter: 'Baseline' or 'Pruned'
        ylimits: Dict of unified y-axis limits
        separate_by_target: If True, use target-specific y-limits
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    df_version = df[df['Version'] == version_filter].copy()
    
    print(f"  Generating plots for version: {version_filter}")
    
    for target_type in ['Price', 'Delta']:
        df_t = df_version[df_version['Target_Type'] == target_type]
        horizons = sorted(df_t['Horizon'].unique())
        
        for horizon in horizons:
            plot_bar_by_horizon(
                csv_file, target_type, horizon, version_filter,
                ylimits, separate_by_target
            )


# For standalone testing
if __name__ == "__main__":
    from plot_utils import compute_unified_ylimits
    
    csv_file = "../experiment_results_clean.csv"
    df = load_csv(csv_file)
    df_baseline = df[df['Version'] == 'Baseline'].copy()
    ylimits = compute_unified_ylimits(df_baseline, 2.5, 3.0, True)
    
    generate_all_plots(csv_file, "Baseline", ylimits, True)
