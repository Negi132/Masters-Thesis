"""
Summary table generator
========================
Produces two tables that summarise findings buried in the raw results
but not currently surfaced in the paper:

  1. Best feature set per (Model, Horizon) for each target.
     Cell format: "{exp_num} ({mae:.1f})"

  2. Pruning summary - which feature groups each (Model, Experiment) pair
     dropped the most columns from, based on permutation importance.

Outputs go to ../Plots/Tables/ as both .tex (for direct LaTeX inclusion
via \\input{}) and .csv (for manual edits or sanity checking).

Run alongside the existing plot scripts; this is not invoked by
master_plotter.py automatically. Call directly:
    python generate_summary_tables.py
"""

import os
import json
import sys
import pandas as pd

# Allow importing config from the parent ML_Pipeline package if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# CONFIGURATION
# =====================================================================
CSV_FILE = "../ML_Pipeline/experiment_results_clean.csv"
CSV_FILE_FALLBACK = "../ML_Pipeline/experiment_results.csv"

# Location of pruned_features_{target}.json files. The permutation
# importance script writes them to the current working dir when run
# from inside ML_Pipeline, so we look there first. Falls back to the
# current working directory if not found.
PRUNED_FEATURES_DIR = "../ML_Pipeline"

# Output directory is a sibling of this script, matching the convention
# used by plot_1_deterioration.py, plot_10_supplementary.py, etc.
OUTPUT_DIR = "Tables"

HORIZONS = [0, 24, 48, 72, 96, 120, 144, 168]
TARGETS = ["Price", "Delta"]
ALL_MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest",
              "LSTM", "GRU", "Transformer", "AutoGluon"]

# Maps base experiment names to a stable short number for display.
# Matches the numbering used in Section V-A of the paper.
EXP_NUMBERS = {
    "Exp1_Weather_Only":                              1,
    "Exp2_Weather_WeatherLags_Only":                  2,
    "Exp3_Weather_Prices":                            3,
    "Exp4_Weather_WeatherLags_Prices":                4,
    "Exp5_Weather_Grid":                              5,
    "Exp6_Weather_WeatherLags_Grid":                  6,
    "Exp7_Weather_Grid_Prices":                       7,
    "Exp8_Weather_WeatherLags_Grid_Prices":           8,
    "Exp9_Weather_Grid_Gridlags":                     9,
    "Exp10_Weather_WeatherLags_Grid_Gridlags":        10,
    "Exp11_Weather_Grid_Gridlags_Prices":             11,
    "Exp12_Weather_WeatherLags_Grid_Gridlags_Prices": 12,
    "Exp13_Total_Information":                        13,
}

# Maps feature groups to (column-name-prefix-patterns). This is how we
# bucket the pruning JSON's flat column list back into the feature
# groups used in the experiments. The patterns are evaluated in order;
# the first match wins. Order matters because some patterns are
# substrings of others (e.g. all WeatherLags columns also start with a
# weather feature name).
#
# IMPORTANT: This mapping is heuristic. It is built to match the column
# naming conventions in Script 9 (build_master_features). If a column
# does not match any pattern, it goes to "Other" and a warning is
# printed so you can extend this map.
FEATURE_GROUP_PATTERNS = [
    # Imputed flags first - unambiguous suffix check before anything else
    ("Imputed_Flag", lambda c: c.endswith("_imputed")),
    # Lagged price columns - look for Price + lag
    ("PriceLags", lambda c: ("SpotPrice" in c or "DeltaPrice" in c)
                           and "_lag_" in c),
    # WeatherLags MUST NOT match grid production cols (which contain MWh).
    # Weather features have lowercase prefixes like avg_*, max_*, min_*,
    # whereas grid features are PascalCase ending in MWh/Power.
    ("WeatherLags", lambda c: "_lag_" in c
                              and any(c.startswith(p) for p in
                                      ["avg_", "max_", "min_", "mean_"])),
    # GridExchangeLags before GridLags - the Exchange* columns are a
    # specific subset of grid columns
    ("GridExchangeLags", lambda c: "_lag_" in c
                                   and ("Exchange" in c or "ExchangeNo" in c)),
    # Other lagged grid columns - MWh or production keywords
    ("GridLags", lambda c: "_lag_" in c
                           and ("MWh" in c or "Power" in c or "Loss" in c)),
    # Time features by exact-name set
    ("Time", lambda c: c in [
        "HourOfDay", "DayOfWeek", "Month", "DayOfYear",
        "HourOfDay_sin", "HourOfDay_cos",
        "DayOfWeek_sin", "DayOfWeek_cos",
        "Month_sin", "Month_cos",
        "DayOfYear_sin", "DayOfYear_cos",
    ]),
    # Spot price columns (no lag)
    ("Prices", lambda c: c in ["SpotPriceEUR", "SpotPriceDKK", "DeltaPrice"]),
    # Exchange features (no lag)
    ("GridExchange", lambda c: ("Exchange" in c or "ExchangeNo" in c)
                               and "_lag_" not in c),
    # Other grid features - MWh or named production columns, no lag
    ("Grid", lambda c: "_lag_" not in c
                       and ("MWh" in c or "Power" in c or "Loss" in c
                            or "Consumption" in c)),
    # Weather features (no lag) - lowercase avg/max/min prefix
    ("Weather", lambda c: any(c.startswith(p) for p in
                              ["avg_", "max_", "min_", "mean_"])
                          and "_lag_" not in c),
]

GROUP_DISPLAY_ORDER = [
    "Weather", "WeatherLags", "Grid", "GridLags",
    "GridExchange", "GridExchangeLags",
    "Prices", "PriceLags", "Time", "Imputed_Flag", "Other"
]


# =====================================================================
# LATEX ESCAPING
# =====================================================================
# Any string heading into a LaTeX cell or header has to escape the
# special characters that LaTeX treats specially in text mode. The
# main offender for our tables is the underscore (`_`) which appears
# in experiment names, column names, and group names like
# "Imputed_Flag". Without escaping, LaTeX will either error out
# ("Missing $ inserted") or render text as garbled subscripts.
_LATEX_SPECIAL_CHARS = {
    '\\': r'\textbackslash{}',
    '{':  r'\{',
    '}':  r'\}',
    '_':  r'\_',
    '%':  r'\%',
    '$':  r'\$',
    '&':  r'\&',
    '#':  r'\#',
    '^':  r'\^{}',
    '~':  r'\~{}',
}

def latex_escape(s):
    """Escape LaTeX special characters in a plain-text string.
    Leaves None/NaN values unchanged (they'll usually appear as "--")."""
    if s is None:
        return ""
    s = str(s)
    # Backslash MUST be escaped first or it would double-escape itself
    out = s.replace('\\', _LATEX_SPECIAL_CHARS['\\'])
    for ch, repl in _LATEX_SPECIAL_CHARS.items():
        if ch == '\\':
            continue
        out = out.replace(ch, repl)
    return out


# =====================================================================
# CSV LOADING (matches the convention used by plot scripts)
# =====================================================================
def load_results():
    csv = CSV_FILE if os.path.exists(CSV_FILE) else CSV_FILE_FALLBACK
    if not os.path.exists(csv):
        print(f"[ERROR] Neither {CSV_FILE} nor {CSV_FILE_FALLBACK} found.")
        return None
    print(f"  Loading: {csv}")
    df = pd.read_csv(csv, sep=None, engine='python')

    if 'Status' in df.columns:
        df = df[df['Status'] == 'SUCCESS'].copy()

    df['Target_Type'] = df['Target'].astype(str).apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(
        lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)

    def clean_exp_name(name):
        s = str(name)
        for tag in ['_0h', '_24h', '_48h', '_72h', '_96h', '_120h', '_144h', '_168h']:
            if tag in s:
                return s.split(tag)[0]
        return s
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)

    # Exclude supplementary experiment rows from baseline tables
    df = df[~df['Experiment'].str.contains(
        'Pruned|FullWeek|Fullweek|Midas|Optuna|MAELoss|DK2|GRUtanh|Naive|'
        'medium_quality|best_quality',
        case=False, na=False)]

    return df


# =====================================================================
# TABLE 1: BEST FEATURE SET PER (MODEL, HORIZON)
# =====================================================================
def build_best_feature_set_table(df, target_type):
    """
    For each (Model, Horizon) cell, find the experiment with the lowest
    MAE for the given target_type. Returns a DataFrame with models as
    rows and horizons as columns. Cells contain: "{exp_num} ({mae:.1f})"
    """
    df_t = df[df['Target_Type'] == target_type].copy()
    if df_t.empty:
        print(f"  [SKIP] No {target_type} data")
        return None

    # For Delta, 0h is not predicted, drop that column from the matrix
    horizons = [h for h in HORIZONS if not (target_type == 'Delta' and h == 0)]

    table = pd.DataFrame(index=ALL_MODELS, columns=[f"{h}h" for h in horizons])

    for model in ALL_MODELS:
        for h in horizons:
            sub = df_t[(df_t['Model'] == model) & (df_t['Horizon'] == h)]
            if sub.empty:
                table.at[model, f"{h}h"] = "--"
                continue
            best_row = sub.loc[sub['MAE'].idxmin()]
            exp_num = EXP_NUMBERS.get(best_row['Base_Experiment'], "?")
            mae = best_row['MAE']
            table.at[model, f"{h}h"] = f"{exp_num} ({mae:.1f})"
    return table


def write_best_feature_set_latex(table, target_type, out_path):
    horizons = list(table.columns)
    ncols = len(horizons)
    col_spec = "l" + "c" * ncols

    lines = []
    lines.append("% Auto-generated by generate_summary_tables.py - do not edit by hand")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Best feature set per model and horizon, target = {target_type}. "
                 f"Cells show experiment number (1-13, see Section V-A) and the corresponding "
                 f"MAE in EUR/MWh. Lower MAE is better.}}")
    lines.append(f"\\label{{tab:best-feature-set-{target_type.lower()}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    header = " & ".join(["\\textbf{Model}"] +
                        [f"\\textbf{{{latex_escape(h)}}}" for h in horizons]) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for model in table.index:
        cells = [latex_escape(model)] + \
                [latex_escape(table.at[model, h]) for h in horizons]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))


# =====================================================================
# TABLE 2: PRUNING DROP COUNTS PER FEATURE GROUP
# =====================================================================
def classify_column(col):
    """Assign a column to one of the feature groups via FEATURE_GROUP_PATTERNS."""
    for group_name, predicate in FEATURE_GROUP_PATTERNS:
        try:
            if predicate(col):
                return group_name
        except Exception:
            continue
    return "Other"


def load_pruned_json(target_type):
    """Returns the pruned_features_{target_type}.json contents, or None."""
    path = os.path.join(PRUNED_FEATURES_DIR, f"pruned_features_{target_type}.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    with open(path, 'r') as f:
        return json.load(f)


def load_full_feature_lists_per_experiment(df, target_type):
    """
    For each base_experiment, return the set of feature columns used by
    that experiment (before pruning).

    Two strategies, tried in order:

      1. EXACT: Import config.COL_GROUPS and look up the columns for each
         group in the experiment's group list. This gives the true
         original feature set. Requires ML_Pipeline to be importable.

      2. APPROXIMATE: Union of kept columns across all models for the
         same base experiment. Columns dropped by ALL models become
         invisible to the table. Used as a fallback when the import
         fails.

    Returns: ({base_exp: set_of_columns}, is_exact: bool)
    """
    # Try the exact strategy first
    try:
        from ML_Pipeline import config
        from ML_Pipeline import data_loader

        # Maps from experiment short name (matches EXP_NUMBERS keys) to
        # the group list, matching BASE_EXPERIMENTS_MAP in script 16.
        BASE_EXPS = {
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

        full_by_exp = {}
        for exp_name, groups in BASE_EXPS.items():
            cols = set()
            for g in groups:
                cols.update(config.COL_GROUPS.get(g, []))
            full_by_exp[exp_name] = cols

        # Verify at least one exp got non-empty cols, otherwise treat
        # this as a failed import (config may be loaded but COL_GROUPS
        # empty for some reason)
        if any(len(v) > 0 for v in full_by_exp.values()):
            print("  [INFO] Pruning table using EXACT feature lists from config.COL_GROUPS")
            return full_by_exp, True
        else:
            print("  [WARN] config.COL_GROUPS is empty - falling back to approximate")
    except Exception as e:
        print(f"  [INFO] Could not import config.COL_GROUPS ({e}). "
              "Falling back to approximate feature reconstruction.")

    # Approximate fallback
    pruned = load_pruned_json(target_type)
    if not pruned:
        return {}, False

    union_by_exp = {}
    for key, kept_cols in pruned.items():
        for model in ALL_MODELS:
            prefix = f"{model}_"
            if key.startswith(prefix):
                exp = key[len(prefix):]
                union_by_exp.setdefault(exp, set()).update(kept_cols)
                break
    return union_by_exp, False


def classify_best_mean_worst_from_json_order(pruned_dict):
    """
    Labels each model's pruned experiments as Best/Mean/Worst based on
    their position in the pruning JSON file.

    Why insertion order: script 16 (permutation_importance.py) builds
    its tasks list in [Best, Mean, Worst] order per model (see lines
    71-75 of script 16), then iterates that list inserting into
    pruned_feature_dict. Python 3.7+ preserves dict insertion order,
    and json.dump preserves it on serialisation. So:

        For each model, the FIRST key written = Best,
                       the SECOND key written = Mean,
                       the THIRD key written  = Worst.

    Edge case: if only 2 entries are present for a model, this means
    script 16's dedup logic collapsed two of the three tasks onto the
    same base experiment - most often because the "Mean" experiment
    (closest to the mean MAE) coincided with "Best". The surviving
    two entries are therefore Best + Worst, not Best + Mean. See the
    n==2 branch below for the labelling rule.

    Args:
        pruned_dict: the raw {key: kept_cols} loaded from JSON, with
                     insertion order preserved.

    Returns: {(Model, normalised_base_exp): 'Best' | 'Mean' | 'Worst'}
    """
    # First pass: collect the first 3 distinct base_exp per model, in
    # insertion order. Per-horizon pruned re-runs that share a base_exp
    # with an earlier entry don't count as new positions.
    positions_per_model = {m: [] for m in ALL_MODELS}

    for key in pruned_dict.keys():
        model_found = None
        raw_after_model = None
        for model in ALL_MODELS:
            prefix = f"{model}_"
            if key.startswith(prefix):
                model_found = model
                raw_after_model = key[len(prefix):]
                break
        if not model_found:
            continue

        base_exp, _ = _normalise_base_exp(raw_after_model)
        if base_exp not in positions_per_model[model_found]:
            positions_per_model[model_found].append(base_exp)

    # Second pass: assign labels based on the count of distinct entries.
    label_map = {}
    for model, base_exps in positions_per_model.items():
        n = len(base_exps)
        if n == 0:
            continue
        elif n == 1:
            label_map[(model, base_exps[0])] = 'Best'
        elif n == 2:
            # Two entries present. Script 16's dedup (lines 93-100 of
            # 16_permutation_importance.py) iterates tasks in
            # [Best, Mean, Worst] order and keeps only the first
            # occurrence per (model, exp). So if Mean happens to equal
            # Best (common when MAEs cluster tightly), Mean is dropped
            # and the surviving entries are Best + Worst in that order.
            # Less commonly Worst could equal Best (only with very
            # degenerate data), in which case the surviving entries
            # would be Best + Mean. We label Best + Worst since the
            # Mean-collapses-onto-Best case is dominant in practice.
            label_map[(model, base_exps[0])] = 'Best'
            label_map[(model, base_exps[1])] = 'Worst'
        else:
            # 3+ entries. First=Best, second=Mean, third=Worst.
            # Any extras beyond the third are left unlabelled.
            label_map[(model, base_exps[0])] = 'Best'
            label_map[(model, base_exps[1])] = 'Mean'
            label_map[(model, base_exps[2])] = 'Worst'

    return label_map


# Standalone helper so both classify_* and build_pruning_table use it
HORIZON_SUFFIXES_STRIP = ['_0h', '_24h', '_48h', '_72h', '_96h',
                          '_120h', '_144h', '_168h']

def _normalise_base_exp(raw):
    """Strip trailing _{H}h_{Price|Delta}_Pruned suffixes from a base
    experiment name. Returns (cleaned_name, had_suffix_flag)."""
    s = raw
    had_suffix = False
    for h in HORIZON_SUFFIXES_STRIP:
        if h in s:
            s = s.split(h)[0]
            had_suffix = True
            break
    if s.endswith('_Pruned'):
        s = s[:-len('_Pruned')]
        had_suffix = True
    return s, had_suffix


def classify_best_mean_worst(df_results, target_type, pruning_keys_by_model):
    """DEPRECATED in favour of classify_best_mean_worst_from_json_order().
    Kept as a stub for backward compatibility - returns empty dict so
    callers fall back to the JSON-order labelling."""
    return {}


def build_pruning_table(target_type):
    """
    Returns (DataFrame, is_exact) where:
      - DataFrame is indexed by (Model, Base_Experiment) with columns
        = feature group names, values = count of columns dropped from
        that group during pruning.
      - is_exact is True if dropped counts were computed against the
        full original feature set (via config.COL_GROUPS), False if
        they were computed against the union of kept columns across
        all models.
    """
    pruned = load_pruned_json(target_type)
    if not pruned:
        return None, False

    df_results = load_results()
    full_by_exp, is_exact = load_full_feature_lists_per_experiment(df_results, target_type)
    if not full_by_exp:
        return None, False

    # The permutation importance JSON keys come in two known shapes:
    #
    #   1. "{Model}_{ExpN_long_name}"
    #      e.g. "CatBoost_Exp4_Weather_WeatherLags_Prices"
    #      This is the original 24h permutation importance run from
    #      script 16, which is what the paper describes in Section V-F.
    #
    #   2. "{Model}_{ExpN_long_name}_{H}h_{Price|Delta}_Pruned"
    #      e.g. "CatBoost_Exp4_Weather_WeatherLags_Prices_120h_Delta_Pruned"
    #      This is a later full-horizon pruned re-run, NOT the 24h
    #      permutation importance pass.
    #
    # The table is intended to show the 24h permutation importance
    # results (per Section V-F). To do this safely we:
    #   a. Strip any trailing horizon/target/Pruned suffixes from the
    #      key so it normalises to the base experiment name.
    #   b. After normalising, deduplicate per (Model, BaseExperiment) by
    #      preferring keys that did NOT carry a suffix (i.e. the original
    #      24h run). If only suffixed keys are present we warn and
    #      include the first one so the row isn't silently lost.
    rows = []
    unrecognized_cols = set()
    seen = {}  # (Model, BaseExp) -> (raw_key, kept_cols, has_suffix)

    # normalise_base_exp() is defined at module scope as _normalise_base_exp.
    # Alias for readability in this function.
    normalise_base_exp = _normalise_base_exp

    for key, kept_cols in pruned.items():
        model_found = None
        raw_after_model = None
        for model in ALL_MODELS:
            prefix = f"{model}_"
            if key.startswith(prefix):
                model_found = model
                raw_after_model = key[len(prefix):]
                break
        if not model_found:
            print(f"  [WARN] Cannot parse model from key '{key}' - skipping")
            continue

        base_exp, had_suffix = normalise_base_exp(raw_after_model)
        dedup_key = (model_found, base_exp)

        # Prefer entries that came from the original 24h run (had_suffix=False)
        # over entries that came from per-horizon pruned re-runs.
        existing = seen.get(dedup_key)
        if existing is None:
            seen[dedup_key] = (key, kept_cols, had_suffix)
        elif existing[2] and not had_suffix:
            # Replace a suffix-bearing entry with a clean one
            seen[dedup_key] = (key, kept_cols, had_suffix)
        # else: keep the existing (already clean or also-suffixed) entry

    # Warn about cases where only suffixed keys were available
    suffixed_only = [(m, b) for (m, b), (_, _, had) in seen.items() if had]
    if suffixed_only:
        print(f"  [WARN] {len(suffixed_only)} (model, experiment) pairs only "
              f"had per-horizon pruned-rerun keys, not original 24h ones. "
              f"Using the suffixed key as a fallback. Example: "
              f"{suffixed_only[0]}")

    # Use the original JSON insertion order to assign Best/Mean/Worst
    # labels. Script 16 writes its tasks in [Best, Mean, Worst] order
    # per model, so the FIRST encountered key per model is Best, the
    # SECOND is Mean, and the THIRD is Worst. This is the only labelling
    # that reliably matches what script 16 actually decided at run time,
    # because the current CSV may not reflect the data state when 16 ran.
    bmw_labels = classify_best_mean_worst_from_json_order(pruned)

    # Diagnostic: report per-model JSON entry counts BEFORE deduplication
    # so the user can spot cases where the JSON itself is incomplete
    # (e.g. a model has only 2 entries instead of 3).
    raw_count_per_model = {m: 0 for m in ALL_MODELS}
    seen_base_per_model = {m: set() for m in ALL_MODELS}
    for key in pruned.keys():
        for model in ALL_MODELS:
            if key.startswith(f"{model}_"):
                raw_count_per_model[model] += 1
                base, _ = normalise_base_exp(key[len(model) + 1:])
                seen_base_per_model[model].add(base)
                break
    print(f"  [INFO] Pruning JSON contents for target={target_type}:")
    for m in ALL_MODELS:
        if raw_count_per_model[m] > 0:
            distinct = len(seen_base_per_model[m])
            print(f"    {m}: {raw_count_per_model[m]} raw keys, "
                  f"{distinct} distinct base experiments")
        else:
            print(f"    {m}: NO pruning data")

    # Now build the table rows from the deduplicated set
    for (model_found, base_exp), (orig_key, kept_cols, _) in seen.items():
        full_set = full_by_exp.get(base_exp, set())
        kept_set = set(kept_cols)
        dropped = full_set - kept_set

        counts = {g: 0 for g in GROUP_DISPLAY_ORDER}
        for col in dropped:
            g = classify_column(col)
            counts[g] = counts.get(g, 0) + 1
            if g == "Other":
                unrecognized_cols.add(col)

        rows.append({
            "Model": model_found,
            "Base_Experiment": base_exp,
            "BMW": bmw_labels.get((model_found, base_exp), 'Unknown'),
            "Total_Kept": len(kept_cols),
            "Total_Dropped": len(dropped),
            **counts,
        })

    if unrecognized_cols:
        print(f"  [INFO] {len(unrecognized_cols)} columns were not matched "
              f"by any feature group pattern. Examples: "
              f"{list(unrecognized_cols)[:5]}")
        print("         These are bucketed as 'Other'. To classify them, extend "
              "FEATURE_GROUP_PATTERNS at the top of this script.")

    return pd.DataFrame(rows), is_exact


def write_pruning_latex(table, target_type, out_path, is_exact=False):
    if table is None or table.empty:
        print(f"  [SKIP] No pruning data for {target_type}")
        return

    # Drop empty group columns to keep the table narrow
    group_cols = [c for c in GROUP_DISPLAY_ORDER if c in table.columns
                  and table[c].sum() > 0]

    col_spec = "ll" + "c" * (len(group_cols) + 2)
    lines = []
    lines.append("% Auto-generated by generate_summary_tables.py - do not edit by hand")
    lines.append("% Requires \\usepackage[table]{xcolor} in the document preamble")
    lines.append("% (the [table] option pulls in colortbl which provides \\rowcolor).")
    lines.append("")
    # Use \providecolor so we don't clash if the document already defines
    # these elsewhere. Pastel green/yellow/red for Best/Mean/Worst rows.
    lines.append("\\providecolor{bmwBest}{RGB}{200,230,201}   % pastel green")
    lines.append("\\providecolor{bmwMean}{RGB}{255,243,205}   % pastel yellow")
    lines.append("\\providecolor{bmwWorst}{RGB}{248,215,218}  % pastel red")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    if is_exact:
        lines.append(f"\\caption{{Feature pruning summary, target = {target_type}. "
                     f"For each (model, base experiment) combination on which "
                     f"permutation importance was performed, the table shows the "
                     f"number of features kept after pruning, the total number "
                     f"dropped, and the breakdown of dropped features by feature "
                     f"group. Higher drop counts in a group indicate that the "
                     f"model derived less value from that group. "
                     f"Row colours indicate which selection bucket the feature set "
                     f"was drawn from: green = best-performing, "
                     f"yellow = mean-performing, red = worst-performing.}}")
    else:
        lines.append(f"\\caption{{Feature pruning summary, target = {target_type}. "
                     f"For each (model, base experiment) combination on which "
                     f"permutation importance was performed, the table shows the "
                     f"number of features kept after pruning and the comparative "
                     f"number dropped, broken down by feature group. A feature is "
                     f"considered dropped by model X if it was kept by at least "
                     f"one other model but not by X (relative pruning measure). "
                     f"Row colours indicate which selection bucket the feature set "
                     f"was drawn from: green = best-performing, "
                     f"yellow = mean-performing, red = worst-performing.}}")
    lines.append(f"\\label{{tab:pruning-{target_type.lower()}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    header = ["\\textbf{Model}", "\\textbf{Base Experiment}",
              "\\textbf{Kept}", "\\textbf{Dropped}"] + \
             [f"\\textbf{{{latex_escape(g)}}}" for g in group_cols]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")

    # Ordering rules:
    #   1. Models follow ALL_MODELS order (CatBoost, LightGBM, XGBoost,
    #      RandomForest, LSTM, GRU, Transformer, AutoGluon).
    #   2. Within each model, rows are in Best -> Mean -> Worst order.
    #      Anything labelled 'Unknown' goes last in experiment-number
    #      order for graceful degradation.
    BMW_ORDER = {'Best': 0, 'Mean': 1, 'Worst': 2, 'Unknown': 3}
    BMW_COLOR = {'Best': 'bmwBest', 'Mean': 'bmwMean',
                 'Worst': 'bmwWorst', 'Unknown': None}
    MODEL_ORDER = {m: i for i, m in enumerate(ALL_MODELS)}

    def sort_key(row):
        model_idx = MODEL_ORDER.get(row['Model'], len(ALL_MODELS))
        bmw_idx = BMW_ORDER.get(row['BMW'], 99)
        exp_idx = EXP_NUMBERS.get(row['Base_Experiment'], 999)
        return (model_idx, bmw_idx, exp_idx)

    sorted_rows = sorted(table.to_dict('records'), key=sort_key)
    last_model = None
    for r in sorted_rows:
        model_cell = r['Model'] if r['Model'] != last_model else ""
        last_model = r['Model']

        # Short experiment label (Exp1, Exp2, ... or raw fallback)
        if r['Base_Experiment'] in EXP_NUMBERS:
            exp_label = f"Exp{EXP_NUMBERS[r['Base_Experiment']]}"
        else:
            exp_label = r['Base_Experiment']

        row_cells = [latex_escape(model_cell),
                     latex_escape(exp_label),
                     str(r['Total_Kept']),
                     str(r['Total_Dropped'])] + \
                    [str(r[g]) for g in group_cols]
        row_str = " & ".join(row_cells) + " \\\\"

        # Prepend the row colour command if a label is known
        colour = BMW_COLOR.get(r['BMW'])
        if colour:
            lines.append(f"\\rowcolor{{{colour}}} {row_str}")
        else:
            lines.append(row_str)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))


# =====================================================================
# MAIN
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  SUMMARY TABLE GENERATOR")
    print("=" * 60)

    df = load_results()
    if df is None:
        return

    for target in TARGETS:
        print(f"\n[Table 1: Best feature set per model+horizon, target={target}]")
        t1 = build_best_feature_set_table(df, target)
        if t1 is not None:
            csv_path = os.path.join(OUTPUT_DIR, f"best_feature_set_{target}.csv")
            tex_path = os.path.join(OUTPUT_DIR, f"best_feature_set_{target}.tex")
            t1.to_csv(csv_path)
            write_best_feature_set_latex(t1, target, tex_path)
            print(f"  Wrote {os.path.basename(csv_path)} and {os.path.basename(tex_path)}")

        print(f"\n[Table 2: Feature pruning drop counts, target={target}]")
        t2, t2_exact = build_pruning_table(target)
        if t2 is not None and not t2.empty:
            csv_path = os.path.join(OUTPUT_DIR, f"pruning_dropped_{target}.csv")
            tex_path = os.path.join(OUTPUT_DIR, f"pruning_dropped_{target}.tex")
            t2.to_csv(csv_path, index=False)
            write_pruning_latex(t2, target, tex_path, is_exact=t2_exact)
            print(f"  Wrote {os.path.basename(csv_path)} and {os.path.basename(tex_path)}")
        else:
            print(f"  [SKIP] No pruning data for {target}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
