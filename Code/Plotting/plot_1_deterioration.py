"""
PLOT 1: FORECAST DETERIORATION
===============================
Shows actual vs predicted values over a time slice at different forecast horizons.
Demonstrates how prediction accuracy degrades as horizon increases.
"""

import os
import glob
import pickle
import matplotlib.pyplot as plt

OUTPUT_DIR = "Plot_1_Deterioration"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded winners per horizon per target (from baseline experiments)
WINNERS_PRICE = {
    0:  "Exp11_Weather_Grid_Gridlags_Prices",
    24: "Exp13_Total_Information",
    48: "Exp13_Total_Information",
    72: "Exp13_Total_Information"
}

WINNERS_DELTA = {
    24: "Exp8_Weather_WeatherLags_Grid_Prices",
    48: "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices",
    72: "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices"
}


def plot_model_deterioration(target_type, model_name, slice_length=336, log_dir="../Experiment_Logs"):
    """
    Plot actual vs predicted values showing forecast deterioration.
    
    Args:
        target_type: 'Price' or 'Delta'
        model_name: Model architecture name (e.g., 'CatBoost', 'LSTM')
        slice_length: Number of hours to display (default: 336 = 2 weeks)
        log_dir: Path to directory containing prediction PKL files
    """
    if not os.path.exists(log_dir):
        print(f"  [ERROR] {log_dir} directory not found.")
        return
    
    winners = WINNERS_PRICE if target_type == "Price" else WINNERS_DELTA
    
    plt.figure(figsize=(15, 7))
    ground_truth_plotted = False
    lines_plotted = 0
    colors = {0: 'blue', 24: 'green', 48: 'orange', 72: 'red'}
    
    for horizon, base_exp in winners.items():
        # Find matching PKL files
        pattern = f"{log_dir}/*{base_exp}*{horizon}h*{target_type}*.pkl"
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            continue
        
        # Prefer baseline over pruned
        matching_files.sort(key=lambda x: 'Pruned' in x, reverse=False)
        
        # Load model data
        model_data = None
        for file_name in matching_files:
            try:
                with open(file_name, 'rb') as f:
                    data = pickle.load(f)
                    if model_name in data:
                        model_data = data[model_name]
                        break
            except Exception as e:
                print(f"  [WARNING] Could not read {file_name}: {e}")
                continue
        
        if model_data is None or len(model_data.get('y_true', [])) == 0:
            continue
        
        # Extract a centered slice of the predictions
        total_length = len(model_data['y_true'])
        if total_length > slice_length:
            actual_start = (total_length // 2) - (slice_length // 2)
            actual_end = actual_start + slice_length
        else:
            actual_start, actual_end = 0, total_length
        
        x_axis = range(actual_end - actual_start)
        
        # Plot ground truth once
        if not ground_truth_plotted:
            plt.plot(
                x_axis, 
                model_data['y_true'][actual_start:actual_end],
                label='Actual Ground Truth', 
                color='black', 
                linewidth=2, 
                linestyle='dashed'
            )
            ground_truth_plotted = True
        
        # Plot predictions for this horizon
        plt.plot(
            x_axis, 
            model_data['y_pred'][actual_start:actual_end],
            label=f'Predicted {horizon}h Ahead', 
            color=colors.get(horizon, 'grey'), 
            alpha=0.7
        )
        lines_plotted += 1
    
    if lines_plotted == 0:
        print(f"  [SKIP] No data for {model_name} ({target_type})")
        plt.close()
        return
    
    plt.title(
        f"Forecast Deterioration over Horizons\n"
        f"(Model: {model_name} | Target: {target_type})", 
        fontsize=14
    )
    plt.xlabel("Hours Elapsed in Test Slice", fontsize=12)
    plt.ylabel(f"Euro value ({target_type})", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, f"Deterioration_Plot_{model_name}_{target_type}.png")
    plt.savefig(save_path, dpi=300)
    print(f"    ✓ {os.path.basename(save_path)}")
    plt.close()


def generate_all_plots(all_models, targets, log_dir="../ML_Pipeline/Experiment_Logs"):
    """
    Generate deterioration plots for all model/target combinations.
    
    Args:
        all_models: List of model names
        targets: List of target types ('Price', 'Delta')
        log_dir: Path to experiment logs directory
    """
    if not os.path.exists(log_dir):
        print(f"  [SKIP] Log directory not found: {log_dir}")
        print(f"  [INFO] Plot 1 requires pickle files from experiment runs")
        return
    
    for target in targets:
        for model in all_models:
            plot_model_deterioration(target, model, slice_length=336, log_dir=log_dir)


# For standalone testing
if __name__ == "__main__":
    models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", 
              "LSTM", "GRU", "Transformer", "AutoGluon"]
    targets = ["Price", "Delta"]
    generate_all_plots(models, targets)
