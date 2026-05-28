"""
MASTER PLOTTER RUNNER
====================
Centralized control for all plotting scripts.
Configure toggles here, then run to generate all enabled plots.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot

from plot_utils import load_csv, compute_unified_ylimits

# =====================================================================
# PLOT TOGGLES - Enable/disable individual plot types
# =====================================================================
RUN_PLOT_1_DETERIORATION    = True
RUN_PLOT_2_VARIANCE_BOX     = True
RUN_PLOT_3_VARIANCE_LINES   = True
RUN_PLOT_4_BAR_BY_HORIZON   = True
RUN_PLOT_5_BAR_BY_MODEL     = True
RUN_PLOT_6_HORIZON_DEGRADE  = True
RUN_PLOT_7_MODEL_COMPARISON = True
RUN_PLOT_8_FEATURE_CONTRIB  = True
RUN_PLOT_9_PRUNING_GAIN     = True
RUN_PLOT_10_SUPPLEMENTARY   = True   # Weekend experiments (Naive, MAE loss, Optuna, etc.)

# =====================================================================
# Y-AXIS SCALING CONFIGURATION
# =====================================================================
# When True: Price and Delta targets get separate y-axis limits
# When False: All plots share one global y-axis limit (both targets together)
SEPARATE_LIMITS_BY_TARGET = True

# Cap multipliers for different plot groups (uses median + IQR × multiplier as cap)
# Groups A+B: Horizon/Model comparison plots (Plots 4, 5, 6, 7) share the same scale
CAP_MULTIPLIER_HORIZON_MODEL = 2.5

# Group C: Variance plots (Plots 2, 3) get their own scale
CAP_MULTIPLIER_VARIANCE = 3.0

# =====================================================================
# DATA SOURCE & LOGS
# =====================================================================
CSV_FILE = "../ML_Pipeline/experiment_results_clean.csv"
LOG_DIR = "../ML_Pipeline/Experiment_Logs"  # For plots 1 & 3 (PKL files)

# =====================================================================
# MODEL AND TARGET DEFINITIONS
# =====================================================================
ALL_MODELS = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", 
              "LSTM", "GRU", "Transformer", "AutoGluon"]
TARGETS = ["Price", "Delta"]


# =====================================================================
# MAIN EXECUTION
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MASTER PLOTTER INITIALIZING")
    print("=" * 70)
    
    # Load CSV once and compute unified y-axis limits
    print(f"\nLoading data from: {CSV_FILE}")
    df_global = load_csv(CSV_FILE)
    if df_global is None:
        print("ERROR: Could not load CSV. Aborting.")
        sys.exit(1)
    
    print(f"  ✓ Loaded {len(df_global)} successful experiment results")
    
    # Filter to baseline only for computing y-axis limits
    df_baseline = df_global[df_global['Version'] == 'Baseline'].copy()
    ylimits = compute_unified_ylimits(
        df_baseline, 
        CAP_MULTIPLIER_HORIZON_MODEL, 
        CAP_MULTIPLIER_VARIANCE,
        SEPARATE_LIMITS_BY_TARGET
    )
    
    print(f"\nY-Axis Scaling Configuration:")
    print(f"  Mode: {'Separate limits per target' if SEPARATE_LIMITS_BY_TARGET else 'Global limit'}")
    print(f"  Horizon/Model plots multiplier: {CAP_MULTIPLIER_HORIZON_MODEL}")
    print(f"  Variance plots multiplier: {CAP_MULTIPLIER_VARIANCE}")
    print()
    
    # ================================================================
    # PLOT 1: FORECAST DETERIORATION
    # ================================================================
    if RUN_PLOT_1_DETERIORATION:
        print("─" * 70)
        print("PLOT 1: Forecast Deterioration")
        print("─" * 70)
        try:
            import plot_1_deterioration
            plot_1_deterioration.generate_all_plots(ALL_MODELS, TARGETS, LOG_DIR)
            print("  ✓ Plot 1 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 1 failed: {e}\n")
    
    # ================================================================
    # PLOT 2: VARIANCE BOXPLOTS
    # ================================================================
    if RUN_PLOT_2_VARIANCE_BOX:
        print("─" * 70)
        print("PLOT 2: Model Variance Boxplots")
        print("─" * 70)
        try:
            import plot_2_variance_box
            plot_2_variance_box.generate_all_plots(
                CSV_FILE, ylimits, SEPARATE_LIMITS_BY_TARGET
            )
            print("  ✓ Plot 2 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 2 failed: {e}\n")
    
    # ================================================================
    # PLOT 3: VARIANCE LINES
    # ================================================================
    if RUN_PLOT_3_VARIANCE_LINES:
        print("─" * 70)
        print("PLOT 3: Feature Sensitivity Lines")
        print("─" * 70)
        try:
            import plot_3_variance_lines
            plot_3_variance_lines.generate_all_plots(
                ALL_MODELS, TARGETS, ylimits, SEPARATE_LIMITS_BY_TARGET, CSV_FILE, LOG_DIR
            )
            print("  ✓ Plot 3 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 3 failed: {e}\n")
    
    # ================================================================
    # PLOT 4: BAR BY HORIZON
    # ================================================================
    if RUN_PLOT_4_BAR_BY_HORIZON:
        print("─" * 70)
        print("PLOT 4: Bar Charts by Horizon")
        print("─" * 70)
        try:
            import plot_4_bar_by_horizon
            for version in ["Baseline", "Pruned"]:
                plot_4_bar_by_horizon.generate_all_plots(
                    CSV_FILE, version, ylimits, SEPARATE_LIMITS_BY_TARGET
                )
            print("  ✓ Plot 4 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 4 failed: {e}\n")
    
    # ================================================================
    # PLOT 5: BAR BY MODEL
    # ================================================================
    if RUN_PLOT_5_BAR_BY_MODEL:
        print("─" * 70)
        print("PLOT 5: Bar Charts by Model")
        print("─" * 70)
        try:
            import plot_5_bar_by_model
            for version in ["Baseline", "Pruned"]:
                plot_5_bar_by_model.generate_all_plots(
                    CSV_FILE, version, ylimits, SEPARATE_LIMITS_BY_TARGET
                )
            print("  ✓ Plot 5 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 5 failed: {e}\n")
    
    # ================================================================
    # PLOT 6: HORIZON DEGRADATION
    # ================================================================
    if RUN_PLOT_6_HORIZON_DEGRADE:
        print("─" * 70)
        print("PLOT 6: Horizon Degradation")
        print("─" * 70)
        try:
            import plot_6_horizon_degradation
            plot_6_horizon_degradation.generate_all_plots(
                CSV_FILE, ylimits, SEPARATE_LIMITS_BY_TARGET
            )
            print("  ✓ Plot 6 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 6 failed: {e}\n")
    
    # ================================================================
    # PLOT 7: MODEL COMPARISON
    # ================================================================
    if RUN_PLOT_7_MODEL_COMPARISON:
        print("─" * 70)
        print("PLOT 7: Model Comparison")
        print("─" * 70)
        try:
            import plot_7_model_comparison
            plot_7_model_comparison.generate_all_plots(
                CSV_FILE, ylimits, SEPARATE_LIMITS_BY_TARGET
            )
            print("  ✓ Plot 7 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 7 failed: {e}\n")
    
    # ================================================================
    # PLOT 8: FEATURE CONTRIBUTION
    # ================================================================
    if RUN_PLOT_8_FEATURE_CONTRIB:
        print("─" * 70)
        print("PLOT 8: Feature Contribution")
        print("─" * 70)
        try:
            import plot_8_feature_contribution
            plot_8_feature_contribution.generate_all_plots(CSV_FILE)
            print("  ✓ Plot 8 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 8 failed: {e}\n")
    
    # ================================================================
    # PLOT 9: PRUNING GAIN
    # ================================================================
    if RUN_PLOT_9_PRUNING_GAIN:
        print("─" * 70)
        print("PLOT 9: Pruning Gain")
        print("─" * 70)
        try:
            import plot_9_pruning_gain
            plot_9_pruning_gain.generate_all_plots(CSV_FILE)
            print("  ✓ Plot 9 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 9 failed: {e}\n")
    
    # ================================================================
    # PLOT 10: SUPPLEMENTARY EXPERIMENTS
    # ================================================================
    if RUN_PLOT_10_SUPPLEMENTARY:
        print("─" * 70)
        print("PLOT 10: Supplementary Experiments")
        print("─" * 70)
        try:
            import plot_10_supplementary
            plot_10_supplementary.generate_all_plots(CSV_FILE)
            print("  ✓ Plot 10 complete\n")
        except Exception as e:
            print(f"  ✗ Plot 10 failed: {e}\n")
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 70)
    print("  ALL ENABLED PLOTS GENERATED")
    print("  Check the subdirectories within /Plotting/ for outputs")
    print("=" * 70)
