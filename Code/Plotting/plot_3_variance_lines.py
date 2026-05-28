"""
PLOT 3: FEATURE SENSITIVITY LINES
==================================
Shows actual vs predicted values for Best/Mean/Worst feature sets.
Demonstrates how feature selection affects prediction quality.
"""

import os
import glob
import pickle
import matplotlib.pyplot as plt
from plot_utils import load_csv, get_best_mean_worst, get_ylimit_for_plot, apply_cap_annotation

OUTPUT_DIR = "Plot_3_Variance_Lines"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_variance_lines(target_type, model_name, horizon, slice_length, ylimits, 
                       separate_by_target, csv_file, log_dir):
    """
    Plot time series showing Best/Mean/Worst feature set performance.
    
    Args:
        target_type: 'Price' or 'Delta'
        model_name: Model architecture name
        horizon: Forecasting horizon in hours
        slice_length: Number of hours to display
        ylimits: Dict of unified y-axis limits
        separate_by_target: If True, use target-specific y-limits
        csv_file: Path to experiment results CSV
        log_dir: Path to experiment logs directory
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # Get Best/Mean/Worst experiments for this model at this horizon
    mask = (df['Target_Type'] == target_type) & (df['Horizon'] == horizon) & \
           (df['Model'] == model_name) & (df['Version'] == 'Baseline')
    plot_data = df[mask].copy()
    
    if plot_data.empty:
        print(f"  [SKIP] No baseline data for {model_name} at {horizon}h ({target_type})")
        return
    
    best_exp, mean_exp, worst_exp = get_best_mean_worst(plot_data)
    
    if best_exp is None:
        print(f"  [SKIP] Could not identify experiments for {model_name}")
        return
    
    experiments_to_load = {
        "Best Model (Min MAE)": best_exp,
        "Average Model (Mean MAE)": mean_exp,
        "Worst Model (Max MAE)": worst_exp
    }
    colors = {
        "Best Model (Min MAE)": "green", 
        "Average Model (Mean MAE)": "orange", 
        "Worst Model (Max MAE)": "red"
    }
    
    for version in ["Baseline", "FullWeek", "Pruned"]:
        fig, ax = plt.subplots(figsize=(15, 7))
        ground_truth_plotted = False
        lines_plotted = 0
        
        for label, base_exp in experiments_to_load.items():
            pattern = f"{log_dir}/*{base_exp}*{horizon}h*{target_type}*.pkl"
            matching_files = glob.glob(pattern)
            
            if not matching_files:
                continue
            
            # Filter by version
            if version == "Pruned":
                filtered = [f for f in matching_files if "Pruned" in f]
            elif version == "FullWeek":
                filtered = [f for f in matching_files if "FullWeek" in f or "Fullweek" in f]
            else:  # Baseline
                filtered = [f for f in matching_files 
                           if "Pruned" not in f and "FullWeek" not in f and "Fullweek" not in f]
            
            if not filtered:
                continue
            
            # Load model data
            model_data = None
            for file_name in filtered:
                try:
                    with open(file_name, 'rb') as f:
                        data = pickle.load(f)
                        if model_name in data:
                            model_data = data[model_name]
                            break
                except Exception:
                    continue
            
            if model_data is None or len(model_data.get('y_true', [])) == 0:
                continue
            
            # Extract slice
            total_length = len(model_data['y_true'])
            if total_length > slice_length:
                start = (total_length // 2) - (slice_length // 2)
                end = start + slice_length
            else:
                start, end = 0, total_length
            
            x_axis = range(end - start)
            
            # Plot ground truth once
            if not ground_truth_plotted:
                ax.plot(
                    x_axis, 
                    model_data['y_true'][start:end],
                    label='Actual Ground Truth', 
                    color='black', 
                    linewidth=2, 
                    linestyle='dashed'
                )
                ground_truth_plotted = True
            
            # Plot predictions
            ax.plot(
                x_axis, 
                model_data['y_pred'][start:end],
                label=label, 
                color=colors[label], 
                alpha=0.7
            )
            lines_plotted += 1
        
        if lines_plotted == 0:
            print(f"  [SKIP] No {version} data for {model_name}")
            plt.close()
            continue
        
        # Apply unified y-axis limit
        if ylimits:
            cap = get_ylimit_for_plot(ylimits, 'variance', target_type, separate_by_target)
            if cap:
                ax.set_ylim(top=cap)
                apply_cap_annotation(ax, cap, 'Price')
        
        ax.set_title(
            f"Feature Set Sensitivity: {model_name} ({version})\n"
            f"(Target: {target_type} | Horizon: {horizon}h)", 
            fontsize=14
        )
        ax.set_xlabel("Hours Elapsed in Test Slice", fontsize=12)
        ax.set_ylabel(f"Euro value ({target_type})", fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(
            OUTPUT_DIR, 
            f"Variance_Lines_{model_name}_{target_type}_{horizon}h_{version}.png"
        )
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


def generate_all_plots(all_models, targets, ylimits, separate_by_target, 
                       csv_file, log_dir="../ML_Pipeline/Experiment_Logs"):
    """
    Generate variance line plots for all model/target combinations at 24h.
    
    Args:
        all_models: List of model names
        targets: List of target types
        ylimits: Dict of unified y-axis limits
        separate_by_target: If True, use target-specific y-limits
        csv_file: Path to experiment results CSV
        log_dir: Path to experiment logs directory
    """
    if not os.path.exists(log_dir):
        print(f"  [SKIP] Log directory not found: {log_dir}")
        print(f"  [INFO] Plot 3 requires pickle files from experiment runs")
        return
    
    for target in targets:
        for model in all_models:
            plot_variance_lines(
                target, model, horizon=24, slice_length=336,
                ylimits=ylimits, separate_by_target=separate_by_target,
                csv_file=csv_file, log_dir=log_dir
            )


# For standalone testing
if __name__ == "__main__":
    from plot_utils import compute_unified_ylimits
    
    models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", 
              "LSTM", "GRU", "Transformer", "AutoGluon"]
    targets = ["Price", "Delta"]
    
    csv_file = "../experiment_results_clean.csv"
    df = load_csv(csv_file)
    df_baseline = df[df['Version'] == 'Baseline'].copy()
    ylimits = compute_unified_ylimits(df_baseline, 2.5, 3.0, True)
    
    generate_all_plots(models, targets, ylimits, True, csv_file)
