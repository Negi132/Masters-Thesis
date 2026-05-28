"""
PLOT 2: MODEL VARIANCE BOXPLOTS
================================
Shows distribution of MAE across feature sets for each model.
Compares Baseline vs Pruned versions.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from plot_utils import load_csv, get_ylimit_for_plot, apply_cap_annotation

OUTPUT_DIR = "Plot_2_Variance_Box"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_feature_variance(csv_file, target_type, horizon, ylimits, separate_by_target, metric="MAE"):
    """
    Generate boxplot showing variance across feature sets per model.
    
    Args:
        csv_file: Path to experiment results CSV
        target_type: 'Price' or 'Delta'
        horizon: Forecasting horizon in hours
        ylimits: Dict of unified y-axis limits
        separate_by_target: If True, use target-specific y-limits
        metric: Evaluation metric to plot (default: MAE)
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    mask = (df['Target_Type'] == target_type) & (df['Horizon'] == horizon)
    plot_data = df[mask].copy()
    
    if plot_data.empty:
        print(f"  [SKIP] No data for {target_type} at {horizon}h")
        return
    
    # Filter outliers for R2 or extreme MAE values
    if metric == 'R2':
        plot_data = plot_data[plot_data[metric] > -1.0]
    else:
        upper_limit = plot_data[metric].quantile(0.95) * 2
        plot_data = plot_data[plot_data[metric] <= upper_limit]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Boxplot with stripplot overlay
    sns.boxplot(
        x='Model', y=metric, hue='Version', data=plot_data, ax=ax,
        palette={'Baseline': 'lightcoral', 'Pruned': 'mediumseagreen', 'FullWeek': 'steelblue'},
        showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"}
    )
    
    sns.stripplot(
        x='Model', y=metric, hue='Version', dodge=True, data=plot_data, ax=ax,
        palette="dark:.25", alpha=0.4, size=3, legend=False, jitter=True
    )
    
    # Apply unified y-axis limit
    cap = None
    if ylimits:
        cap = get_ylimit_for_plot(ylimits, 'variance', target_type, separate_by_target)
        if cap:
            ax.set_ylim(top=cap)
            apply_cap_annotation(ax, cap, metric)
    
    ax.set_title(
        f"Model Variance: Baseline vs Pruned Features\n"
        f"(Target: {target_type} | Horizon: {horizon}h | Metric: {metric})", 
        fontsize=14
    )
    ax.set_xlabel("Model Architecture", fontsize=12)
    ax.set_ylabel(f"{metric} Score (Lower is better)", fontsize=12)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(title="Data Version", loc='upper right')
    plt.tight_layout()
    
    save_path = os.path.join(
        OUTPUT_DIR, 
        f"Variance_Box_Split_{target_type}_{horizon}h_{metric}.png"
    )
    plt.savefig(save_path, dpi=300)
    print(f"    ✓ {os.path.basename(save_path)}")
    plt.close()


def generate_all_plots(csv_file, ylimits, separate_by_target):
    """
    Generate variance boxplots for Delta@24h and Price@0h.
    
    Args:
        csv_file: Path to experiment results CSV
        ylimits: Dict of unified y-axis limits
        separate_by_target: If True, use target-specific y-limits
    """
    # Standard variance plots per original design
    plot_feature_variance(csv_file, "Delta", 24, ylimits, separate_by_target)
    plot_feature_variance(csv_file, "Price", 0, ylimits, separate_by_target)


# For standalone testing
if __name__ == "__main__":
    from plot_utils import compute_unified_ylimits
    
    csv_file = "../experiment_results_clean.csv"
    df = load_csv(csv_file)
    df_baseline = df[df['Version'] == 'Baseline'].copy()
    ylimits = compute_unified_ylimits(df_baseline, 2.5, 3.0, True)
    
    generate_all_plots(csv_file, ylimits, True)
