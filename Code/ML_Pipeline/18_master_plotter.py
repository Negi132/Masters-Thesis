import os
import pickle
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# PLOT TOGGLES - Set to True/False to control what gets generated
# =====================================================================
RUN_PLOT_1_DETERIORATION    = True
RUN_PLOT_2_VARIANCE_BOX     = True
RUN_PLOT_3_VARIANCE_LINES   = True
RUN_PLOT_4_BAR_BY_HORIZON   = True   # Best/Mean/Worst per model, one plot per horizon
RUN_PLOT_5_BAR_BY_MODEL     = True   # Best/Mean/Worst per horizon, one plot per model
RUN_PLOT_6_HORIZON_DEGRADE  = True   # MAE vs horizon per model — the "how fast does it degrade" plot
RUN_PLOT_7_MODEL_COMPARISON = True   # Best MAE per model per horizon — the "who wins" plot
RUN_PLOT_8_FEATURE_CONTRIB  = True   # MAE improvement as feature groups are added progressively
RUN_PLOT_9_PRUNING_GAIN     = True   # % MAE improvement from Baseline to Pruned per model

# =====================================================================
# BAR PLOT Y-AXIS CAP
# Bars exceeding this multiple of the median value are capped and
# annotated with their true value so outliers don't crush the scale.
# Set to None to disable capping entirely.
# =====================================================================
BAR_YAXIS_CAP_MULTIPLIER = 3.0

# =====================================================================
# PATH CONFIGURATION
# =====================================================================
PARENT_DIR       = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
MASTER_PLOTS_DIR = os.path.join(PARENT_DIR, "Plots")

DIR_DETERIORATION  = os.path.join(MASTER_PLOTS_DIR, "Deterioration")
DIR_VARIANCE_BOX   = os.path.join(MASTER_PLOTS_DIR, "Variance_Box")
DIR_VARIANCE_LINES = os.path.join(MASTER_PLOTS_DIR, "Variance_Lines")
DIR_BAR_HORIZON    = os.path.join(MASTER_PLOTS_DIR, "Bar_By_Horizon")
DIR_BAR_MODEL      = os.path.join(MASTER_PLOTS_DIR, "Bar_By_Model")
DIR_HORIZON_DEGRADE  = os.path.join(MASTER_PLOTS_DIR, "Horizon_Degradation")
DIR_MODEL_COMPARISON = os.path.join(MASTER_PLOTS_DIR, "Model_Comparison")
DIR_FEATURE_CONTRIB  = os.path.join(MASTER_PLOTS_DIR, "Feature_Contribution")
DIR_PRUNING_GAIN     = os.path.join(MASTER_PLOTS_DIR, "Pruning_Gain")

for directory in [DIR_DETERIORATION, DIR_VARIANCE_BOX, DIR_VARIANCE_LINES,
                  DIR_BAR_HORIZON, DIR_BAR_MODEL, DIR_HORIZON_DEGRADE,
                  DIR_MODEL_COMPARISON, DIR_FEATURE_CONTRIB, DIR_PRUNING_GAIN]:
    os.makedirs(directory, exist_ok=True)


# =====================================================================
# SHARED HELPER: Load and prepare CSV
# =====================================================================
def load_csv(csv_file="experiment_results.csv"):
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
# SHARED HELPER: Apply y-axis cap to bar axes
# =====================================================================
def apply_yaxis_cap(ax, all_values, metric):
    """
    Caps the y-axis at BAR_YAXIS_CAP_MULTIPLIER × median of valid values.
    Bars that exceed the cap get a red annotation showing their true value.
    Returns the cap value used, or None if capping was disabled/unnecessary.
    """
    if BAR_YAXIS_CAP_MULTIPLIER is None:
        return None

    valid = [v for v in all_values if v is not None and not (v != v)]  # filter NaN
    if not valid:
        return None

    median_val = sorted(valid)[len(valid) // 2]
    cap        = median_val * BAR_YAXIS_CAP_MULTIPLIER

    if max(valid) <= cap:
        return None  # No capping needed

    ax.set_ylim(top=cap)
    ax.annotate(
        f"Note: {metric} axis capped at {cap:.0f} (={BAR_YAXIS_CAP_MULTIPLIER}× median).\n"
        f"Values above cap shown as red labels at top of bar.",
        xy=(0.01, 0.97), xycoords='axes fraction',
        fontsize=7, va='top', color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea', edgecolor='#c0392b', alpha=0.8)
    )
    return cap


# =====================================================================
# PLOT 1: FORECAST DETERIORATION
# =====================================================================
def plot_model_deterioration(target_type="Price", model_name="CatBoost", slice_length=336):
    log_dir = "Experiment_Logs"
    if not os.path.exists(log_dir):
        print(f"Error: {log_dir} directory not found.")
        return

    if target_type == "Price":
        winners = {
            0:  "Exp11_Weather_Grid_Gridlags_Prices",
            24: "Exp13_Total_Information",
            48: "Exp13_Total_Information",
            72: "Exp13_Total_Information"
        }
    else:
        winners = {
            24: "Exp8_Weather_WeatherLags_Grid_Prices",
            48: "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices",
            72: "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices"
        }

    plt.figure(figsize=(15, 7))
    ground_truth_plotted = False
    lines_plotted = 0
    colors = {0: 'blue', 24: 'green', 48: 'orange', 72: 'red'}

    for horizon, base_exp in winners.items():
        matching_files = glob.glob(f"{log_dir}/*{base_exp}*{horizon}h*{target_type}*.pkl")
        if not matching_files:
            continue

        matching_files.sort(key=lambda x: 'Pruned' in x, reverse=True)

        model_data = None
        for file_name in matching_files:
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
                if model_name in data:
                    model_data = data[model_name]
                    break

        if model_data is None or len(model_data['y_true']) == 0:
            continue

        total_length = len(model_data['y_true'])
        if total_length > slice_length:
            actual_start = (total_length // 2) - (slice_length // 2)
            actual_end   = actual_start + slice_length
        else:
            actual_start, actual_end = 0, total_length

        x_axis = range(actual_end - actual_start)

        if not ground_truth_plotted:
            plt.plot(x_axis, model_data['y_true'][actual_start:actual_end],
                     label='Actual Ground Truth', color='black', linewidth=2, linestyle='dashed')
            ground_truth_plotted = True

        plt.plot(x_axis, model_data['y_pred'][actual_start:actual_end],
                 label=f'Predicted {horizon}h Ahead', color=colors.get(horizon, 'grey'), alpha=0.7)
        lines_plotted += 1

    if lines_plotted == 0:
        print(f"  [WARNING] No data for {model_name} ({target_type}).")
        plt.close()
        return

    plt.title(f"Forecast Deterioration over Horizons\n(Model: {model_name} | Target: {target_type})", fontsize=14)
    plt.xlabel("Hours Elapsed in Test Slice", fontsize=12)
    plt.ylabel(f"Euro value ({target_type})", fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(DIR_DETERIORATION, f"Deterioration_Plot_{model_name}_{target_type}.png")
    plt.savefig(save_path, dpi=300)
    print(f"  [SUCCESS] Deterioration Plot saved: {os.path.basename(save_path)}")
    plt.close()


# =====================================================================
# PLOT 2: MODEL VARIANCE BOXPLOTS
# =====================================================================
def plot_feature_variance(csv_file="experiment_results.csv", target_type="Delta", horizon=24, metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return

    mask = (df['Target_Type'] == target_type) & (df['Horizon'] == horizon)
    plot_data = df[mask].copy()

    if plot_data.empty:
        print(f"  [WARNING] No data for Target: {target_type} at Horizon: {horizon}h.")
        return

    if metric == 'R2':
        plot_data = plot_data[plot_data[metric] > -1.0]
    else:
        upper_limit = plot_data[metric].quantile(0.95) * 2
        plot_data = plot_data[plot_data[metric] <= upper_limit]

    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Model', y=metric, hue='Version', data=plot_data,
                palette={'Baseline': 'lightcoral', 'Pruned': 'mediumseagreen', 'FullWeek': 'steelblue'},
                showmeans=True,
                meanprops={"marker":"o","markerfacecolor":"white","markeredgecolor":"black","markersize":"8"})
    sns.stripplot(x='Model', y=metric, hue='Version', dodge=True, data=plot_data,
                  palette="dark:.25", alpha=0.4, size=3, legend=False, jitter=True)

    plt.title(f"Model Variance: Baseline vs Pruned Features\n(Target: {target_type} | Horizon: {horizon}h | Metric: {metric})", fontsize=14)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel(f"{metric} Score (Lower is better)", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(title="Data Version", loc='upper right')
    plt.tight_layout()
    save_path = os.path.join(DIR_VARIANCE_BOX, f"Variance_Box_Split_{target_type}_{horizon}h_{metric}.png")
    plt.savefig(save_path, dpi=300)
    print(f"  [SUCCESS] Variance Boxplot saved: {os.path.basename(save_path)}")
    plt.close()


# =====================================================================
# PLOT 3: FEATURE SENSITIVITY LINES
# =====================================================================
def plot_variance_lines(csv_file="experiment_results.csv", target_type="Price",
                        model_name="CatBoost", horizon=24, slice_length=336):
    log_dir = "Experiment_Logs"
    df = load_csv(csv_file)
    if df is None:
        return

    mask = (df['Target_Type'] == target_type) & (df['Horizon'] == horizon) & \
           (df['Model'] == model_name) & (df['Version'] == 'Baseline')
    plot_data = df[mask].copy()

    if plot_data.empty:
        print(f"  [WARNING] No baseline data for {model_name} at {horizon}h ({target_type}).")
        return

    best_exp  = plot_data.loc[plot_data['MAE'].idxmin(), 'Base_Experiment']
    worst_exp = plot_data.loc[plot_data['MAE'].idxmax(), 'Base_Experiment']
    plot_data['MAE_Diff'] = abs(plot_data['MAE'] - plot_data['MAE'].mean())
    mean_exp  = plot_data.sort_values('MAE_Diff').iloc[0]['Base_Experiment']

    experiments_to_load = {
        "Best Model (Min MAE)":    best_exp,
        "Average Model (Mean MAE)": mean_exp,
        "Worst Model (Max MAE)":   worst_exp
    }
    colors = {"Best Model (Min MAE)": "green", "Average Model (Mean MAE)": "orange", "Worst Model (Max MAE)": "red"}

    for version in ["Baseline", "FullWeek", "Pruned"]:
        plt.figure(figsize=(15, 7))
        ground_truth_plotted = False
        lines_plotted = 0

        for label, base_exp in experiments_to_load.items():
            matching_files = glob.glob(f"{log_dir}/*{base_exp}*{horizon}h*{target_type}*.pkl")
            if not matching_files:
                continue

            if version == "Pruned":
                filtered = [f for f in matching_files if "Pruned" in f]
            elif version == "FullWeek":
                filtered = [f for f in matching_files if "FullWeek" in f or "Fullweek" in f]
            else:
                filtered = [f for f in matching_files if "Pruned" not in f and "FullWeek" not in f and "Fullweek" not in f]

            if not filtered:
                continue

            model_data = None
            for file_to_load in filtered:
                with open(file_to_load, 'rb') as f:
                    data = pickle.load(f)
                    if model_name in data:
                        model_data = data[model_name]
                        break

            if model_data is None or len(model_data['y_true']) == 0:
                print(f"  [MISSING DATA] {model_name} in {base_exp} ({version})")
                continue

            total_length = len(model_data['y_true'])
            if total_length > slice_length:
                actual_start = (total_length // 2) - (slice_length // 2)
                actual_end   = actual_start + slice_length
            else:
                actual_start, actual_end = 0, total_length

            x_axis = range(actual_end - actual_start)

            if not ground_truth_plotted:
                plt.plot(x_axis, model_data['y_true'][actual_start:actual_end],
                         label='Actual Ground Truth', color='black', linewidth=2, linestyle='dashed')
                ground_truth_plotted = True

            plt.plot(x_axis, model_data['y_pred'][actual_start:actual_end],
                     label=f'{label} [{base_exp}]', color=colors[label], alpha=0.7)
            lines_plotted += 1

        if lines_plotted == 0:
            plt.close()
            continue

        plt.title(f"{model_name} Feature Sensitivity - {version.upper()}\nHorizon: {horizon}h | Target: {target_type}", fontsize=14)
        plt.xlabel("Hours Elapsed in Test Slice", fontsize=12)
        plt.ylabel(f"Euro value ({target_type})", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(DIR_VARIANCE_LINES, f"Variance_Lines_{model_name}_{target_type}_{horizon}h_{version}.png")
        plt.savefig(save_path, dpi=300)
        print(f"  [SUCCESS] {version} Lines Plot saved for {model_name}")
        plt.close()


# =====================================================================
# PLOT 4: BAR CHART - BEST/MEAN/WORST PER MODEL, ONE PLOT PER HORIZON
# All model types side by side. One plot per horizon, per target type.
# =====================================================================
def plot_bar_by_horizon(csv_file="experiment_results.csv", metric="MAE", version_filter="Baseline"):
    df = load_csv(csv_file)
    if df is None:
        return

    df = df[df['Version'] == version_filter].copy()

    bar_colors = {'Best': '#2ecc71', 'Mean': '#f39c12', 'Worst': '#e74c3c'}
    bar_width   = 0.25
    x_offsets   = {'Best': -bar_width, 'Mean': 0, 'Worst': bar_width}

    for target_type in ["Price", "Delta"]:
        df_t = df[df['Target_Type'] == target_type]
        horizons = sorted(df_t['Horizon'].unique())

        for horizon in horizons:
            df_h = df_t[df_t['Horizon'] == horizon]
            if df_h.empty:
                continue

            models = sorted(df_h['Model'].unique())
            x_positions = np.arange(len(models))

            fig, ax = plt.subplots(figsize=(14, 7))

            for rank_label in ['Best', 'Mean', 'Worst']:
                values = []
                for model in models:
                    model_data = df_h[df_h['Model'] == model][metric].dropna()
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
                    edgecolor='white'
                )

                # Add value labels — place above bar or at cap line if truncated
                for bar, val in zip(bars, values):
                    if not np.isnan(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=7, rotation=45
                        )

            # Collect all plotted values for cap calculation
            all_vals = []
            for rank_label in ['Best', 'Mean', 'Worst']:
                for model in models:
                    v = df_h[df_h['Model'] == model][metric].dropna()
                    if not v.empty:
                        all_vals.append(v.min() if rank_label == 'Best' else
                                        v.max() if rank_label == 'Worst' else v.mean())

            cap = apply_yaxis_cap(ax, all_vals, metric)

            # Annotate bars that were truncated by the cap
            if cap is not None:
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
            plt.tight_layout()

            save_path = os.path.join(
                DIR_BAR_HORIZON,
                f"Bar_ByHorizon_{target_type}_{horizon}h_{version_filter}_{metric}.png"
            )
            plt.savefig(save_path, dpi=300)
            print(f"  [SUCCESS] Bar-by-Horizon saved: {os.path.basename(save_path)}")
            plt.close()


# =====================================================================
# PLOT 5: BAR CHART - BEST/MEAN/WORST PER HORIZON, ONE PLOT PER MODEL
# Shows how each model degrades across horizons. One plot per model, per target.
# =====================================================================
def plot_bar_by_model(csv_file="experiment_results.csv", metric="MAE", version_filter="Baseline"):
    df = load_csv(csv_file)
    if df is None:
        return

    df = df[df['Version'] == version_filter].copy()

    bar_colors = {'Best': '#2ecc71', 'Mean': '#f39c12', 'Worst': '#e74c3c'}
    bar_width   = 0.25
    x_offsets   = {'Best': -bar_width, 'Mean': 0, 'Worst': bar_width}

    for target_type in ["Price", "Delta"]:
        df_t = df[df['Target_Type'] == target_type]
        models = sorted(df_t['Model'].unique())

        for model_name in models:
            df_m = df_t[df_t['Model'] == model_name]
            if df_m.empty:
                continue

            horizons    = sorted(df_m['Horizon'].unique())
            x_positions = np.arange(len(horizons))
            x_labels    = [f"{h}h" for h in horizons]

            fig, ax = plt.subplots(figsize=(14, 7))

            for rank_label in ['Best', 'Mean', 'Worst']:
                values = []
                for horizon in horizons:
                    horizon_data = df_m[df_m['Horizon'] == horizon][metric].dropna()
                    if horizon_data.empty:
                        values.append(np.nan)
                        continue

                    if rank_label == 'Best':
                        values.append(horizon_data.min())
                    elif rank_label == 'Worst':
                        values.append(horizon_data.max())
                    else:
                        values.append(horizon_data.mean())

                bars = ax.bar(
                    x_positions + x_offsets[rank_label],
                    values,
                    width=bar_width,
                    label=rank_label,
                    color=bar_colors[rank_label],
                    alpha=0.85,
                    edgecolor='white'
                )

                # Value labels on top of bars
                for bar, val in zip(bars, values):
                    if not np.isnan(val):
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=7, rotation=45
                        )

            # Collect all plotted values for cap calculation
            all_vals = []
            for rank_label in ['Best', 'Mean', 'Worst']:
                for horizon in horizons:
                    v = df_m[df_m['Horizon'] == horizon][metric].dropna()
                    if not v.empty:
                        all_vals.append(v.min() if rank_label == 'Best' else
                                        v.max() if rank_label == 'Worst' else v.mean())

            cap = apply_yaxis_cap(ax, all_vals, metric)

            # Annotate bars truncated by the cap
            if cap is not None:
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
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.set_xlabel("Forecast Horizon", fontsize=12)
            ax.set_ylabel(f"{metric} (Lower is better)", fontsize=12)
            ax.set_title(
                f"Best / Mean / Worst {metric} Across Horizons\n"
                f"Model: {model_name} | Target: {target_type} | Version: {version_filter}",
                fontsize=14
            )
            ax.legend(title="Feature Set Performance", fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(
                DIR_BAR_MODEL,
                f"Bar_ByModel_{model_name}_{target_type}_{version_filter}_{metric}.png"
            )
            plt.savefig(save_path, dpi=300)
            print(f"  [SUCCESS] Bar-by-Model saved: {os.path.basename(save_path)}")
            plt.close()



# =====================================================================
# PLOT 6: HORIZON DEGRADATION — MAE vs Horizon per model
# One plot per target. Shows how each model's best MAE degrades as
# the forecast horizon increases. Uses best-performing feature set
# per model per horizon from Baseline data.
# =====================================================================
def plot_horizon_degradation(csv_file="experiment_results.csv", metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return

    df_baseline = df[df['Version'] == 'Baseline'].copy()
    tree_models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest"]
    nn_models   = ["LSTM", "GRU", "Transformer", "AutoGluon"]

    model_styles = {
        "CatBoost":     ("#e74c3c", "o",  "-"),
        "LightGBM":     ("#e67e22", "s",  "-"),
        "XGBoost":      ("#f1c40f", "^",  "-"),
        "RandomForest": ("#2ecc71", "D",  "-"),
        "LSTM":         ("#3498db", "o",  "--"),
        "GRU":          ("#9b59b6", "s",  "--"),
        "Transformer":  ("#1abc9c", "^",  "--"),
        "AutoGluon":    ("#e91e63", "D",  "--"),
    }

    for target_type in ["Price", "Delta"]:
        df_t     = df_baseline[df_baseline['Target_Type'] == target_type]
        horizons = sorted(df_t['Horizon'].unique())

        # Split into two subplots: Trees (top) and NNs (bottom)
        # NNs may have outlier values at some horizons that would crush the tree scale
        fig, (ax_tree, ax_nn) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            f"Forecast Accuracy Degradation Across Horizons\n"
            f"Target: {target_type} | Baseline | Best feature set per model",
            fontsize=14
        )

        for ax, model_group, group_label in [
            (ax_tree, tree_models, "Tree Models"),
            (ax_nn,   nn_models,   "Neural Network Models")
        ]:
            group_vals = []
            for model_name in model_group:
                df_m = df_t[df_t['Model'] == model_name]
                if df_m.empty:
                    continue

                best_per_horizon = df_m.groupby('Horizon')[metric].min().reset_index()
                best_per_horizon = best_per_horizon.sort_values('Horizon')

                color, marker, linestyle = model_styles.get(model_name, ('grey', 'o', '-'))
                ax.plot(
                    best_per_horizon['Horizon'],
                    best_per_horizon[metric],
                    label=model_name,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2,
                    markersize=7
                )
                group_vals.extend(best_per_horizon[metric].tolist())

            # Apply cap per subplot independently so trees aren't crushed by NN outliers
            apply_yaxis_cap(ax, group_vals, metric)

            ax.set_ylabel(f"Best {metric} (Lower is better)", fontsize=11)
            ax.set_title(group_label, fontsize=11, pad=4)
            ax.legend(title="Model", fontsize=9, ncol=2)
            ax.grid(True, alpha=0.3)

        ax_nn.set_xticks(horizons)
        ax_nn.set_xticklabels([f"{h}h" for h in horizons], fontsize=10)
        ax_nn.set_xlabel("Forecast Horizon", fontsize=12)

        plt.tight_layout()
        save_path = os.path.join(DIR_HORIZON_DEGRADE,
                                 f"Horizon_Degradation_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"  [SUCCESS] Horizon Degradation saved: {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# PLOT 7: MODEL COMPARISON — Best MAE per model per horizon
# One plot per target. Head-to-head comparison of all models at their
# best feature set, shown as a grouped bar chart per horizon.
# =====================================================================
def plot_model_comparison(csv_file="experiment_results.csv", metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return

    df_baseline = df[df['Version'] == 'Baseline'].copy()
    all_models  = ["CatBoost", "LightGBM", "XGBoost", "RandomForest",
                   "LSTM", "GRU", "Transformer", "AutoGluon"]

    model_colors = {
        "CatBoost":     "#e74c3c",
        "LightGBM":     "#e67e22",
        "XGBoost":      "#f1c40f",
        "RandomForest": "#2ecc71",
        "LSTM":         "#3498db",
        "GRU":          "#9b59b6",
        "Transformer":  "#1abc9c",
        "AutoGluon":    "#e91e63",
    }

    for target_type in ["Price", "Delta"]:
        df_t     = df_baseline[df_baseline['Target_Type'] == target_type]
        horizons = sorted(df_t['Horizon'].unique())

        n_models    = len(all_models)
        bar_width   = 0.8 / n_models
        x_positions = np.arange(len(horizons))

        fig, ax = plt.subplots(figsize=(16, 7))

        for j, model_name in enumerate(all_models):
            df_m = df_t[df_t['Model'] == model_name]
            values = []

            for horizon in horizons:
                h_data = df_m[df_m['Horizon'] == horizon][metric].dropna()
                values.append(h_data.min() if not h_data.empty else np.nan)

            offset = (j - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_positions + offset,
                values,
                width=bar_width,
                label=model_name,
                color=model_colors.get(model_name, 'grey'),
                alpha=0.85,
                edgecolor='white'
            )

            # Add value labels — only if bar is wide enough to be readable
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.15,
                        f'{val:.1f}',
                        ha='center', va='bottom',
                        fontsize=5.5, rotation=90, color='#333333'
                    )

        # Apply y-axis cap
        all_vals = []
        for model_name in all_models:
            df_m = df_t[df_t['Model'] == model_name]
            for horizon in horizons:
                h_data = df_m[df_m['Horizon'] == horizon][metric].dropna()
                if not h_data.empty:
                    all_vals.append(h_data.min())
        apply_yaxis_cap(ax, all_vals, metric)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{h}h" for h in horizons], fontsize=10)
        ax.set_xlabel("Forecast Horizon", fontsize=12)
        ax.set_ylabel(f"Best {metric} Achieved (Lower is better)", fontsize=12)
        ax.set_title(
            f"Model Head-to-Head Comparison — Best {metric} per Horizon\n"
            f"Target: {target_type} | Baseline | Each model at its best feature set",
            fontsize=14
        )
        ax.legend(title="Model", fontsize=9, ncol=4, loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(DIR_MODEL_COMPARISON,
                                 f"Model_Comparison_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"  [SUCCESS] Model Comparison saved: {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# PLOT 8: FEATURE GROUP CONTRIBUTION
# Shows mean MAE across models as feature groups are added progressively.
# Uses the natural ordering of experiments (Exp1 → Exp13).
# One plot per target per horizon.
# =====================================================================
def plot_feature_contribution(csv_file="experiment_results.csv", metric="MAE",
                               horizons_to_plot=None):
    df = load_csv(csv_file)
    if df is None:
        return

    df_baseline = df[df['Version'] == 'Baseline'].copy()

    # Ordered experiment progression — each adds a meaningful group
    PROGRESSION = [
        ("Exp1_Weather_Only",                              "Weather Only\n(Exp1)"),
        ("Exp2_Weather_WeatherLags_Only",                  "Weather\n+ W.Lags\n(Exp2)"),
        ("Exp3_Weather_Prices",                            "Weather\n+ Prices\n(Exp3)"),
        ("Exp4_Weather_WeatherLags_Prices",                "Weather\n+ W.Lags\n+ Prices\n(Exp4)"),
        ("Exp5_Weather_Grid",                              "Weather\n+ Grid\n(Exp5)"),
        ("Exp7_Weather_Grid_Prices",                       "Weather\n+ Grid\n+ Prices\n(Exp7)"),
        ("Exp9_Weather_Grid_Gridlags",                     "Weather\n+ Grid\n+ G.Lags\n(Exp9)"),
        ("Exp11_Weather_Grid_Gridlags_Prices",             "Weather\n+ Grid\n+ G.Lags\n+ Prices\n(Exp11)"),
        ("Exp12_Weather_WeatherLags_Grid_Gridlags_Prices", "Weather\n+ All Lags\n+ Prices\n(Exp12)"),
        ("Exp13_Total_Information",                        "Total\nInformation\n(Exp13)"),
    ]
    exp_names  = [p[0] for p in PROGRESSION]
    exp_labels = [p[1] for p in PROGRESSION]

    if horizons_to_plot is None:
        horizons_to_plot = [24, 48, 96, 168]

    for target_type in ["Price", "Delta"]:
        df_t = df_baseline[df_baseline['Target_Type'] == target_type]

        fig, ax = plt.subplots(figsize=(16, 8))
        colors  = plt.cm.viridis(np.linspace(0, 0.85, len(horizons_to_plot)))

        for color, horizon in zip(colors, horizons_to_plot):
            df_h   = df_t[df_t['Horizon'] == horizon]
            values = []

            for exp in exp_names:
                exp_data = df_h[df_h['Base_Experiment'] == exp][metric].dropna()
                values.append(exp_data.mean() if not exp_data.empty else np.nan)

            ax.plot(
                range(len(exp_names)),
                values,
                label=f"{horizon}h horizon",
                color=color,
                marker='o',
                linewidth=2,
                markersize=6
            )

        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_labels, rotation=0, ha='center', fontsize=8)
        ax.set_xlabel("Feature Group Added", fontsize=12)
        ax.set_ylabel(f"Mean {metric} Across Models (Lower is better)", fontsize=12)
        ax.set_title(
            f"Feature Group Contribution to Forecast Accuracy\n"
            f"Target: {target_type} | Baseline | Mean across all models",
            fontsize=14
        )
        ax.legend(title="Horizon", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(DIR_FEATURE_CONTRIB,
                                 f"Feature_Contribution_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"  [SUCCESS] Feature Contribution saved: {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# PLOT 9: PRUNING GAIN
# Shows % MAE improvement from Baseline to Pruned for each model.
# One plot per target type. Uses best MAE per model in each version.
# =====================================================================
def plot_pruning_gain(csv_file="experiment_results.csv", metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return

    all_models = ["CatBoost", "LightGBM", "XGBoost", "RandomForest",
                  "LSTM", "GRU", "Transformer", "AutoGluon"]

    for target_type in ["Price", "Delta"]:
        df_t = df[df['Target_Type'] == target_type]

        horizons     = sorted(df_t['Horizon'].unique())
        x_positions  = np.arange(len(horizons))
        bar_width    = 0.8 / len(all_models)

        fig, ax = plt.subplots(figsize=(18, 7))
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

        # Draw vertical separator lines between horizon groups
        for sep in range(1, len(horizons)):
            ax.axvline(x=sep - 0.5, color='grey', linewidth=0.8,
                       linestyle=':', alpha=0.6)

        model_colors = {
            "CatBoost":     "#e74c3c", "LightGBM":     "#e67e22",
            "XGBoost":      "#f1c40f", "RandomForest": "#2ecc71",
            "LSTM":         "#3498db", "GRU":          "#9b59b6",
            "Transformer":  "#1abc9c", "AutoGluon":    "#e91e63",
        }

        any_data = False
        for j, model_name in enumerate(all_models):
            gains = []

            for horizon in horizons:
                base_data = df_t[
                    (df_t['Version'] == 'Baseline') &
                    (df_t['Model'] == model_name) &
                    (df_t['Horizon'] == horizon)
                ][metric].dropna()

                pruned_data = df_t[
                    (df_t['Version'] == 'Pruned') &
                    (df_t['Model'] == model_name) &
                    (df_t['Horizon'] == horizon)
                ][metric].dropna()

                if base_data.empty or pruned_data.empty:
                    gains.append(np.nan)
                    continue

                base_best   = base_data.min()
                pruned_best = pruned_data.min()

                # Positive = improvement, negative = pruning made things worse
                pct_gain = ((base_best - pruned_best) / base_best) * 100
                gains.append(pct_gain)

            if all(np.isnan(g) for g in gains):
                continue

            any_data = True
            offset = (j - len(all_models) / 2 + 0.5) * bar_width
            bars = ax.bar(
                x_positions + offset,
                gains,
                width=bar_width,
                label=model_name,
                color=model_colors.get(model_name, 'grey'),
                alpha=0.85,
                edgecolor='white',
                linewidth=0.5
            )

        if not any_data:
            print(f"  [WARNING] No Pruned data found for {target_type}. Skipping pruning gain plot.")
            plt.close()
            continue

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{h}h" for h in horizons], fontsize=10)
        ax.set_xlabel("Forecast Horizon", fontsize=12)
        ax.set_ylabel("% MAE Improvement (Positive = Pruning Helped)", fontsize=12)
        ax.set_title(
            f"Pruning Engine Gain: Baseline → Pruned\n"
            f"Target: {target_type} | Positive bars = pruning improved accuracy",
            fontsize=14
        )
        ax.legend(title="Model", fontsize=9, ncol=4, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(DIR_PRUNING_GAIN,
                                 f"Pruning_Gain_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"  [SUCCESS] Pruning Gain saved: {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("==================================================")
    print(f"  MASTER PLOTTER INITIALIZING")
    print(f"  Output Directory: {MASTER_PLOTS_DIR}")
    print("==================================================\n")

    all_models  = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "LSTM", "GRU", "Transformer", "AutoGluon"]
    targets     = ["Price", "Delta"]

    if RUN_PLOT_1_DETERIORATION:
        print("--- 1. Generating Deterioration Plots ---")
        for target in targets:
            for model in all_models:
                plot_model_deterioration(target_type=target, model_name=model, slice_length=336)

    if RUN_PLOT_2_VARIANCE_BOX:
        print("\n--- 2. Generating Model Variance Boxplots ---")
        plot_feature_variance(target_type="Delta", horizon=24, metric="MAE")
        plot_feature_variance(target_type="Price", horizon=0,  metric="MAE")

    if RUN_PLOT_3_VARIANCE_LINES:
        print("\n--- 3. Generating Feature Sensitivity Lines ---")
        for target in targets:
            for model in all_models:
                plot_variance_lines(target_type=target, model_name=model, horizon=24, slice_length=336)

    if RUN_PLOT_4_BAR_BY_HORIZON:
        print("\n--- 4. Generating Bar Charts by Horizon ---")
        for version in ["Baseline", "Pruned"]:
            plot_bar_by_horizon(metric="MAE", version_filter=version)

    if RUN_PLOT_5_BAR_BY_MODEL:
        print("\n--- 5. Generating Bar Charts by Model ---")
        for version in ["Baseline", "Pruned"]:
            plot_bar_by_model(metric="MAE", version_filter=version)

    if RUN_PLOT_6_HORIZON_DEGRADE:
        print("\n--- 6. Generating Horizon Degradation Plots ---")
        plot_horizon_degradation(metric="MAE")

    if RUN_PLOT_7_MODEL_COMPARISON:
        print("\n--- 7. Generating Model Comparison Plots ---")
        plot_model_comparison(metric="MAE")

    if RUN_PLOT_8_FEATURE_CONTRIB:
        print("\n--- 8. Generating Feature Contribution Plots ---")
        plot_feature_contribution(metric="MAE")

    if RUN_PLOT_9_PRUNING_GAIN:
        print("\n--- 9. Generating Pruning Gain Plots ---")
        plot_pruning_gain(metric="MAE")

    print("\n==================================================")
    print("  ALL SELECTED PLOTS GENERATED!")
    print(f"  Check the '{MASTER_PLOTS_DIR}' folder.")
    print("==================================================")