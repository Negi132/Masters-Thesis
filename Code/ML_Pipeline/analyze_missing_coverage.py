"""
COVERAGE ANALYSIS SCRIPT (v3 - Collapse Detection)
====================================================
Detects combinations where best/mean/worst all resolve to the same
experiment (due to only 1-2 experiments being present), making bar
plots meaningless. 

To determine WHICH experiments to queue, it uses the 24h horizon as a
reference — where all 13 experiments were run for all models — and uses
those MAE rankings to identify which experiments are genuinely best,
mean and worst for each model. Those are then queued for the missing
horizons.
"""

import os
import glob
import pickle
import pandas as pd

CSV_FILE = "experiment_results.csv"
LOG_DIR  = "Experiment_Logs"

BASE_EXPERIMENTS_MAP = {
    "Exp1_Weather_Only":                              ["Weather", "Time"],
    "Exp2_Weather_WeatherLags_Only":                  ["Weather", "WeatherLags", "Time"],
    "Exp3_Weather_Prices":                            ["Weather", "Prices", "Time"],
    "Exp4_Weather_WeatherLags_Prices":                ["Weather", "WeatherLags", "Prices", "Time"],
    "Exp5_Weather_Grid":                              ["Weather", "Grid", "GridExchange", "Time"],
    "Exp6_Weather_WeatherLags_Grid":                  ["Weather", "WeatherLags", "Grid", "GridExchange", "Time"],
    "Exp7_Weather_Grid_Prices":                       ["Weather", "Grid", "GridExchange", "Prices", "Time"],
    "Exp8_Weather_WeatherLags_Grid_Prices":           ["Weather", "WeatherLags", "Grid", "GridExchange", "Prices", "Time"],
    "Exp9_Weather_Grid_Gridlags":                     ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"],
    "Exp10_Weather_WeatherLags_Grid_Gridlags":        ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Time"],
    "Exp11_Weather_Grid_Gridlags_Prices":             ["Weather", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"],
    "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices": ["Weather", "WeatherLags", "Grid", "GridExchange", "GridLags", "GridExchangeLags", "Prices", "Time"],
    "Exp13_Total_Information":                        ["All_Features"],
}

# =====================================================================
# HELPERS
# =====================================================================

def clean_exp_name(name):
    base = str(name)
    for tag in ['_0h','_24h','_48h','_72h','_96h','_120h','_144h','_168h']:
        if tag in base:
            base = base.split(tag)[0]
            break
    return base


def get_best_mean_worst_from_df(df_subset):
    """
    Returns (best_exp, mean_exp, worst_exp) from a filtered DataFrame.
    Returns (None, None, None) if empty.
    """
    if df_subset.empty:
        return None, None, None

    best_exp  = df_subset.loc[df_subset['MAE'].idxmin(), 'Base_Experiment']
    worst_exp = df_subset.loc[df_subset['MAE'].idxmax(), 'Base_Experiment']
    mean_mae  = df_subset['MAE'].mean()
    df_copy   = df_subset.copy()
    df_copy['MAE_Diff'] = abs(df_copy['MAE'] - mean_mae)
    mean_exp  = df_copy.sort_values('MAE_Diff').iloc[0]['Base_Experiment']

    return best_exp, mean_exp, worst_exp


def has_pkl_data(base_exp, horizon, target_type, model_name):
    """Returns True if a baseline pkl file exists with non-empty data for this model."""
    pattern  = f"{LOG_DIR}/*{base_exp}*{horizon}h*{target_type}*.pkl"
    matches  = glob.glob(pattern)
    baseline = [f for f in matches
                if "Pruned" not in f and "FullWeek" not in f and "Fullweek" not in f]

    for f in baseline:
        try:
            with open(f, 'rb') as fh:
                data = pickle.load(fh)
            if model_name in data and len(data[model_name].get('y_true', [])) > 0:
                return True
        except Exception:
            continue
    return False


# =====================================================================
# MAIN
# =====================================================================

def main():
    if not os.path.exists(CSV_FILE):
        print(f"[ERROR] {CSV_FILE} not found.")
        return

    df = pd.read_csv(CSV_FILE, sep=None, engine='python')
    df = df[df['Status'] == 'SUCCESS'].copy()
    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    df['Version'] = df['Experiment'].apply(
        lambda x: 'Pruned'   if 'Pruned'   in str(x) else
                 ('FullWeek' if ('FullWeek' in str(x) or 'Fullweek' in str(x)) else 'Baseline'))

    df_baseline = df[df['Version'] == 'Baseline'].copy()

    print("=" * 75)
    print("  COVERAGE ANALYSIS REPORT (Collapse Detection)")
    print("  Finds combinations where Best = Mean = Worst (meaningless plots)")
    print("=" * 75)

    # gaps[(exp, horizon, target)] -> {info dict}
    gaps    = {}
    n_ok    = 0
    n_flat  = 0
    n_empty = 0

    for target_type in ["Price", "Delta"]:
        df_t     = df_baseline[df_baseline['Target_Type'] == target_type]
        models   = sorted(df_t['Model'].unique())
        horizons = sorted(df_t['Horizon'].unique())

        print(f"\n--- TARGET: {target_type} ---")

        for model_name in models:
            for horizon in horizons:
                df_mh = df_t[
                    (df_t['Model'] == model_name) &
                    (df_t['Horizon'] == horizon)
                ].copy()

                if df_mh.empty:
                    n_empty += 1
                    continue

                best_exp, mean_exp, worst_exp = get_best_mean_worst_from_df(df_mh)
                unique_exps = len({best_exp, mean_exp, worst_exp} - {None})

                # -------------------------------------------------------
                # COLLAPSE DETECTED: all three point to the same experiment
                # -------------------------------------------------------
                if unique_exps < 3:
                    n_flat += 1
                    print(f"  [FLAT]    {model_name:<15} | {target_type:<6} | {horizon}h "
                          f"| Only {df_mh['Base_Experiment'].nunique()} unique exp(s) "
                          f"-> collapsed to [{best_exp}]")

                    # Use 24h rankings as the reference to find true best/mean/worst
                    # for this model (where we have full 13-experiment coverage)
                    df_ref = df_t[
                        (df_t['Model'] == model_name) &
                        (df_t['Horizon'] == 24)
                    ].copy()

                    if df_ref.empty:
                        # Fallback: use overall rankings across all horizons
                        df_ref = df_t[df_t['Model'] == model_name].copy()

                    ref_best, ref_mean, ref_worst = get_best_mean_worst_from_df(df_ref)

                    # Queue the reference experiments that are missing at this horizon
                    for label, exp in [("Best", ref_best),
                                       ("Mean", ref_mean),
                                       ("Worst", ref_worst)]:
                        if exp is None:
                            continue
                        # Skip if this experiment already has pkl data here
                        if has_pkl_data(exp, horizon, target_type, model_name):
                            continue
                        # Skip if it's the experiment already present
                        # (no point queueing what we have)
                        if exp == best_exp and unique_exps == 1:
                            continue

                        key = (exp, horizon, target_type)
                        print(f"            -> Queue {label}: {exp}")

                        if key not in gaps:
                            gaps[key] = {
                                "experiment":       exp,
                                "horizon":          horizon,
                                "target":           target_type,
                                "groups":           BASE_EXPERIMENTS_MAP.get(exp, ["All_Features"]),
                                "needed_by_models": []
                            }
                        if model_name not in gaps[key]["needed_by_models"]:
                            gaps[key]["needed_by_models"].append(model_name)
                else:
                    n_ok += 1

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"  OK (distinct best/mean/worst):  {n_ok}")
    print(f"  FLAT (collapsed to 1 exp):      {n_flat}")
    print(f"  EMPTY (no csv data):            {n_empty}")
    print(f"  Unique experiment runs to queue: {len(gaps)}")
    print("=" * 75)

    if not gaps:
        print("\n  No collapsed combinations found. All plots are meaningful.")
        return

    print("\n  Experiments to run:")
    print("-" * 75)
    for (exp, horizon, target), info in sorted(gaps.items()):
        print(f"  {exp} | {horizon}h | {target}")
        print(f"    Models: {', '.join(info['needed_by_models'])}")

    rows = []
    for info in gaps.values():
        rows.append({
            "experiment":       info["experiment"],
            "horizon":          info["horizon"],
            "target":           info["target"],
            "groups":           str(info["groups"]),
            "needed_by_models": str(info["needed_by_models"])
        })

    pd.DataFrame(rows).to_csv("missing_coverage_queue.csv", index=False)

    # ---------------------------------------------------------------
    # TRAINING RUN BREAKDOWN
    # ---------------------------------------------------------------
    print("\n--- TRAINING RUN BREAKDOWN ---")
    total_model_runs = 0
    model_run_counts = {}

    for info in gaps.values():
        for model in info["needed_by_models"]:
            total_model_runs += 1
            model_run_counts[model] = model_run_counts.get(model, 0) + 1

    print(f"  Queue entries (unique exp × horizon × target): {len(gaps)}")
    print(f"  Total individual model training runs:          {total_model_runs}")
    print(f"\n  Breakdown by model:")
    for model, count in sorted(model_run_counts.items()):
        print(f"    {model:<16}: {count} runs")

    print("\n  Queue saved to: missing_coverage_queue.csv")
    print("  Run run_missing_coverage.py to train these experiments.")


if __name__ == "__main__":
    main()