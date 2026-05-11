import os
import glob
import pickle
import pandas as pd

# =====================================================================
# CONFIGURATION - Match these to your master plotter settings
# =====================================================================
LOG_DIR = "Experiment_Logs"
CSV_FILE = "experiment_results.csv"
HORIZON = 24

ALL_MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "LSTM", "GRU", "Transformer", "AutoGluon"]
TARGETS = ["Price", "Delta"]

# These versions require scripts 15 and 17 - we skip them in this audit
SKIP_VERSIONS = ["Pruned", "FullWeek"]

# Hardcoded winners from the plotter (Plot 1 - Deterioration)
DETERIORATION_WINNERS = {
    "Price": {0: "Exp11_Weather_Grid_Gridlags_Prices", 24: "Exp13_Total_Information",
              48: "Exp13_Total_Information", 72: "Exp13_Total_Information"},
    "Delta": {24: "Exp8_Weather_WeatherLags_Grid_Prices",
              48: "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices",
              72: "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices"}
}

# =====================================================================
# HELPERS
# =====================================================================

def find_baseline_file_with_model(base_exp, horizon, target_type, model_name):
    """Finds a BASELINE pkl file containing the specified model."""
    pattern = f"{LOG_DIR}/*{base_exp}*{horizon}h*{target_type}*.pkl"
    matching = glob.glob(pattern)
    # Baseline only - exclude Pruned and FullWeek
    baseline = [f for f in matching if "Pruned" not in f and "FullWeek" not in f and "Fullweek" not in f]

    for f in baseline:
        try:
            with open(f, 'rb') as fh:
                data = pickle.load(fh)
            if model_name in data and len(data[model_name].get('y_true', [])) > 0:
                return f
        except Exception:
            continue
    return None

def get_best_mean_worst(df, model_name, target_type, horizon):
    """Replicates the plotter's logic for identifying best/mean/worst experiments."""
    mask = (
        (df['Target_Type'] == target_type) &
        (df['Horizon'] == horizon) &
        (df['Model'] == model_name) &
        (~df['Experiment'].str.contains('FullWeek|Fullweek|Pruned', case=False, na=False))
    )
    plot_data = df[mask].copy()

    if plot_data.empty:
        return None, None, None

    def clean_exp_name(name):
        base_exp = str(name)
        for split_str in ['_0h','_24h','_48h','_72h','_96h','_120h','_144h','_168h']:
            if split_str in base_exp:
                base_exp = base_exp.split(split_str)[0]
                break
        return base_exp

    plot_data['Base_Experiment'] = plot_data['Experiment'].apply(clean_exp_name)

    best_exp  = plot_data.loc[plot_data['MAE'].idxmin(), 'Base_Experiment']
    worst_exp = plot_data.loc[plot_data['MAE'].idxmax(), 'Base_Experiment']
    mean_mae  = plot_data['MAE'].mean()
    plot_data['MAE_Diff'] = abs(plot_data['MAE'] - mean_mae)
    mean_exp  = plot_data.sort_values('MAE_Diff').iloc[0]['Base_Experiment']

    return best_exp, mean_exp, worst_exp

# =====================================================================
# MAIN AUDIT
# =====================================================================
def run_audit():
    print("=" * 70)
    print("  MASTER PLOTTER DATA VALIDATION REPORT")
    print("  (Only checking BASELINE data - Pruned/FullWeek skipped)")
    print("=" * 70)

    missing_items = []

    # --- Load CSV ---
    try:
        df = pd.read_csv(CSV_FILE, sep=None, engine='python')
        df = df[df['Status'] == 'SUCCESS'].copy()
        df['Target_Type'] = df['Target'].astype(str).apply(
            lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
        df['Horizon'] = df['Target'].astype(str).apply(
            lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        return

    # ---------------------------------------------------------------
    # PLOT 1: DETERIORATION
    # ---------------------------------------------------------------
    print("\n--- PLOT 1: DETERIORATION ---")
    for target, horizons in DETERIORATION_WINNERS.items():
        for horizon, base_exp in horizons.items():
            for model in ALL_MODELS:
                found = find_baseline_file_with_model(base_exp, horizon, target, model)
                if found:
                    print(f"  [OK]      {model:<15} | {target:<6} | {horizon}h | {base_exp}")
                else:
                    msg = f"  [MISSING] {model:<15} | {target:<6} | {horizon}h | {base_exp}"
                    print(msg)
                    missing_items.append({
                        "Plot": "Deterioration",
                        "Model": model,
                        "Target": target,
                        "Horizon": horizon,
                        "Experiment": base_exp
                    })

    # ---------------------------------------------------------------
    # PLOT 3: VARIANCE LINES (BASELINE only)
    # ---------------------------------------------------------------
    print("\n--- PLOT 3: VARIANCE LINES (BASELINE only) ---")
    for target in TARGETS:
        for model in ALL_MODELS:
            best_exp, mean_exp, worst_exp = get_best_mean_worst(df, model, target, HORIZON)

            if best_exp is None:
                print(f"  [NO CSV]  {model:<15} | {target:<6} | {HORIZON}h | No baseline data in CSV")
                continue

            for label, exp in [("Best", best_exp), ("Mean", mean_exp), ("Worst", worst_exp)]:
                found = find_baseline_file_with_model(exp, HORIZON, target, model)
                if found:
                    print(f"  [OK]      {model:<15} | {target:<6} | {HORIZON}h | {label:<6} | {exp}")
                else:
                    msg = f"  [MISSING] {model:<15} | {target:<6} | {HORIZON}h | {label:<6} | {exp}"
                    print(msg)
                    missing_items.append({
                        "Plot": "Variance_Lines_Baseline",
                        "Model": model,
                        "Target": target,
                        "Horizon": HORIZON,
                        "Experiment": exp
                    })

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    if not missing_items:
        print("  ALL BASELINE DATA PRESENT. Ready to run scripts 16, 17, and 15.")
    else:
        print(f"  MISSING {len(missing_items)} ENTRIES. Experiments to rerun:")
        print("-" * 70)

        # Group by experiment + horizon + target to show minimal rerun set
        rerun_set = {}
        for item in missing_items:
            key = (item['Experiment'], item['Horizon'], item['Target'])
            if key not in rerun_set:
                rerun_set[key] = []
            rerun_set[key].append(item['Model'])

        for (exp, h, t), models in sorted(rerun_set.items()):
            print(f"  Experiment: {exp} | Horizon: {h}h | Target: {t}")
            print(f"    Missing models: {', '.join(models)}")

    print("=" * 70)

if __name__ == "__main__":
    run_audit()