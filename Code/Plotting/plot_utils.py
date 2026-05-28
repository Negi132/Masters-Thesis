"""
Shared utilities for all plotting scripts.
Contains CSV loading, data processing, and unified y-axis scaling logic.
"""

import os
import pandas as pd
import numpy as np

# =====================================================================
# CSV LOADING AND PROCESSING
# =====================================================================
def load_csv(csv_file="experiment_results_clean.csv"):
    """
    Loads and preprocesses the experiment results CSV.
    Adds Target_Type, Horizon, Base_Experiment, and Version columns.
    Filters to SUCCESS status only.
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return None

    try:
        df = pd.read_csv(csv_file, sep=None, engine='python')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    if 'Status' in df.columns:
        df = df[df['Status'] == 'SUCCESS'].copy()

    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)

    def clean_exp_name(name):
        base = str(name)
        for tag in ['_0h','_24h','_48h','_72h','_96h','_120h','_144h','_168h']:
            if tag in base:
                base = base.split(tag)[0]
                break
        return base

    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    df['Version'] = df['Experiment'].apply(
        lambda x: 'Pruned' if 'Pruned' in str(x) else
                  ('FullWeek' if ('FullWeek' in str(x) or 'Fullweek' in str(x)) else 'Baseline'))

    return df


# =====================================================================
# UNIFIED Y-AXIS SCALING
# =====================================================================
def compute_unified_ylimits(df_baseline, cap_multiplier_horizon_model, cap_multiplier_variance, 
                           separate_by_target):
    """
    Computes unified y-axis limits for plot groups to enable visual comparison.
    
    Args:
        df_baseline: DataFrame filtered to Baseline version only
        cap_multiplier_horizon_model: Multiplier for horizon/model plots (Groups A+B)
        cap_multiplier_variance: Multiplier for variance plots (Group C)
        separate_by_target: If True, compute separate limits for Price vs Delta
    
    Returns:
        dict with keys like:
            ('horizon_model', 'Price'): cap_value
            ('horizon_model', 'Delta'): cap_value
            ('horizon_model', 'Global'): cap_value  (if separate_by_target=False)
            ('variance', 'Price'): cap_value
            ('variance', 'Delta'): cap_value
            ('variance', 'Global'): cap_value
    """
    limits = {}
    
    # Group A+B: Horizon and Model comparison plots (4, 5, 6, 7)
    for target in ['Price', 'Delta']:
        df_t = df_baseline[df_baseline['Target_Type'] == target].copy()
        mae_values = df_t['MAE'].dropna().values
        
        if len(mae_values) > 0:
            q1 = np.percentile(mae_values, 25)
            q3 = np.percentile(mae_values, 75)
            iqr = q3 - q1
            median = np.median(mae_values)
            cap = median + (iqr * cap_multiplier_horizon_model)
            limits[('horizon_model', target)] = cap
        
    # If separate_by_target is False, compute one global limit
    if not separate_by_target:
        all_mae = df_baseline['MAE'].dropna().values
        if len(all_mae) > 0:
            q1 = np.percentile(all_mae, 25)
            q3 = np.percentile(all_mae, 75)
            iqr = q3 - q1
            median = np.median(all_mae)
            cap = median + (iqr * cap_multiplier_horizon_model)
            limits[('horizon_model', 'Global')] = cap
    
    # Group C: Variance plots (2, 3) - only at 24h
    df_24h = df_baseline[df_baseline['Horizon'] == 24].copy()
    
    for target in ['Price', 'Delta']:
        df_t = df_24h[df_24h['Target_Type'] == target].copy()
        mae_values = df_t['MAE'].dropna().values
        
        if len(mae_values) > 0:
            q1 = np.percentile(mae_values, 25)
            q3 = np.percentile(mae_values, 75)
            iqr = q3 - q1
            median = np.median(mae_values)
            cap = median + (iqr * cap_multiplier_variance)
            limits[('variance', target)] = cap
    
    if not separate_by_target:
        all_mae_24h = df_24h['MAE'].dropna().values
        if len(all_mae_24h) > 0:
            q1 = np.percentile(all_mae_24h, 25)
            q3 = np.percentile(all_mae_24h, 75)
            iqr = q3 - q1
            median = np.median(all_mae_24h)
            cap = median + (iqr * cap_multiplier_variance)
            limits[('variance', 'Global')] = cap
    
    return limits


def get_ylimit_for_plot(ylimits, plot_group, target_type, separate_by_target):
    """
    Retrieves the appropriate y-axis limit for a given plot.
    
    Args:
        ylimits: dict returned by compute_unified_ylimits()
        plot_group: 'horizon_model' or 'variance'
        target_type: 'Price', 'Delta', or None (for mixed plots)
        separate_by_target: If True, use target-specific limits
    
    Returns:
        float: the y-axis cap to use, or None if not available
    """
    if separate_by_target and target_type:
        return ylimits.get((plot_group, target_type))
    else:
        return ylimits.get((plot_group, 'Global'))


def apply_cap_annotation(ax, cap, metric='MAE'):
    """
    Adds a visual annotation to indicate the plot has been capped.
    
    Args:
        ax: matplotlib axis object
        cap: the cap value applied
        metric: the metric name being displayed
    """
    ax.annotate(
        f"Y-axis capped at {cap:.0f} for cross-plot consistency.\n"
        f"Values exceeding cap shown as labels.",
        xy=(0.01, 0.97), xycoords='axes fraction',
        fontsize=7, va='top', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea', 
                 edgecolor='#c0392b', alpha=0.8)
    )


# =====================================================================
# EXPERIMENT SELECTION HELPERS
# =====================================================================
def get_best_mean_worst(df_subset, metric='MAE'):
    """
    Identifies the best, mean, and worst performing experiments in a dataset.
    
    Args:
        df_subset: DataFrame filtered to specific model/horizon/target
        metric: Metric to rank by (default: MAE)
    
    Returns:
        tuple: (best_exp, mean_exp, worst_exp) or (None, None, None) if empty
    """
    if df_subset.empty:
        return None, None, None

    best_exp = df_subset.loc[df_subset[metric].idxmin(), 'Base_Experiment']
    worst_exp = df_subset.loc[df_subset[metric].idxmax(), 'Base_Experiment']
    
    mean_mae = df_subset[metric].mean()
    df_subset_copy = df_subset.copy()
    df_subset_copy['MAE_Diff'] = abs(df_subset_copy[metric] - mean_mae)
    mean_exp = df_subset_copy.sort_values('MAE_Diff').iloc[0]['Base_Experiment']

    return best_exp, mean_exp, worst_exp
