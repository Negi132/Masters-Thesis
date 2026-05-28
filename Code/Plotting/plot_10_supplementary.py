"""
PLOT 10: SUPPLEMENTARY EXPERIMENTS
===================================
All weekend supplementary experiment visualizations:
  A. Naive Baseline Comparison
  B. Loss Function Alignment (MAE vs MSE training)
  C. Hyperparameter Tuning Gain (Optuna)
  D. AutoGluon Quality Tiers
  E. Weather Data Source (DMI vs Midas)
  F. DK2 Generalization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import load_csv, font_bar_value, bottom_legend_kwargs

OUTPUT_DIR = "Plot_10_Supplementary"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# A. NAIVE BASELINE COMPARISON
# =====================================================================
def plot_naive_baseline_comparison(csv_file):
    """
    Compare best ML model performance vs naive persistence baseline.
    Shows that ML models beat the trivial baseline.
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # Get naive baseline results - filter by Model column (more reliable)
    df_naive = df[df['Model'] == 'Naive_Persistence'].copy()
    
    # Get best ML performance (baseline, non-naive, non-special)
    df_ml = df[
        (df['Version'] == 'Baseline') & 
        (df['Model'] != 'Naive_Persistence') &
        (~df['Experiment'].str.contains(
            'Midas|Optuna|MAELoss|DK2|GRUtanh|Naive',
            case=False, na=False))
    ].copy()
    
    if df_naive.empty:
        print("  [SKIP] Naive baseline data not found")
        return
    
    if df_ml.empty:
        print("  [SKIP] ML baseline data not found")
        return
    
    print(f"  [INFO] Naive rows: {len(df_naive)}, ML rows: {len(df_ml)}")
    
    for target_type in ['Price', 'Delta']:
        df_n = df_naive[df_naive['Target_Type'] == target_type]
        df_m = df_ml[df_ml['Target_Type'] == target_type]
        
        if df_n.empty or df_m.empty:
            print(f"  [SKIP] No {target_type} data")
            continue
        
        # Use horizons that exist in BOTH datasets
        naive_horizons = set(df_n['Horizon'].unique())
        ml_horizons = set(df_m['Horizon'].unique())
        horizons = sorted(naive_horizons & ml_horizons)
        
        if not horizons:
            print(f"  [SKIP] No matching horizons for {target_type}")
            continue
        
        naive_maes = []
        ml_maes = []
        best_ml_models = []
        
        for h in horizons:
            naive_h_data = df_n[df_n['Horizon'] == h]['MAE']
            ml_h_data = df_m[df_m['Horizon'] == h]
            
            if naive_h_data.empty or ml_h_data.empty:
                continue
            
            naive_h = naive_h_data.min()
            best_ml_row = ml_h_data.loc[ml_h_data['MAE'].idxmin()]
            ml_h = best_ml_row['MAE']
            
            naive_maes.append(naive_h)
            ml_maes.append(ml_h)
            best_ml_models.append(best_ml_row['Model'])
        
        if not naive_maes:
            continue
        
        fig, ax = plt.subplots(figsize=(13, 7))
        
        x = np.arange(len(horizons))
        width = 0.35
        
        bars_naive = ax.bar(x - width/2, naive_maes, width, label='Naive Persistence', 
               color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=0.5)
        bars_ml = ax.bar(x + width/2, ml_maes, width, label='Best ML Model',
               color='#2ecc71', alpha=0.85, edgecolor='white', linewidth=0.5)
        
        # Extend the y-axis ceiling to leave room for the percentage labels
        # above each pair of bars. Without this, the labels clip off the top.
        max_bar = max(max(naive_maes), max(ml_maes))
        ax.set_ylim(top=max_bar * 1.18)
        
        # Add value labels and best model name
        for i, (n, m, best_model) in enumerate(zip(naive_maes, ml_maes, best_ml_models)):
            ax.text(i - width/2, n, f'{n:.1f}', ha='center', va='bottom', fontsize=font_bar_value(), fontweight='bold')
            ax.text(i + width/2, m, f'{m:.1f}', ha='center', va='bottom', fontsize=font_bar_value(), fontweight='bold')
            
            # Show best model name inside the bar
            ax.text(i + width/2, m * 0.5, best_model, ha='center', va='center', 
                   fontsize=font_bar_value(), color='white', fontweight='bold', rotation=90)
            
            # Show improvement percentage above both bars, centered in the
            # reserved headroom so it cannot clip the top of the plot.
            if n > 0:
                improvement = ((n - m) / n) * 100
                color = 'green' if improvement > 0 else 'red'
                symbol = '↓' if improvement > 0 else '↑'
                ax.text(i, max(n, m) + (max_bar * 0.09), f'{symbol}{abs(improvement):.0f}%', 
                       ha='center', va='center', fontsize=font_bar_value(), color=color, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h}h' for h in horizons])
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('MAE (Lower is better)')
        ax.set_title(f'ML Models vs Naive Persistence Baseline\nTarget: {target_type}')
        ax.legend(**bottom_legend_kwargs(ncol=2))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'Naive_Baseline_Comparison_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# B. LOSS FUNCTION ALIGNMENT (MAE vs MSE Training)
# =====================================================================
def plot_loss_function_comparison(csv_file):
    """
    Compare MAE-trained vs MSE-trained neural networks across all horizons.
    
    Output: One figure per (model, target) combination, showing
    Best/Mean/Worst MAE for both MSE and MAE training across all horizons.
    Layout mirrors Bar_ByModel plots from the main pipeline.
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # MAE-trained models (from weekend/extended experiments)
    df_mae = df[df['Experiment'].str.contains('MAELoss', case=False, na=False)].copy()
    
    if df_mae.empty:
        print("  [SKIP] MAE loss function data not found")
        return
    
    # Find which models actually have MAE loss data
    tested_models = sorted(df_mae['Model'].unique())
    print(f"  [INFO] MAE loss tested for: {tested_models}")
    
    # MSE-trained models (baseline)
    df_mse = df[
        (df['Version'] == 'Baseline') & 
        (df['Model'].isin(tested_models)) &
        (~df['Experiment'].str.contains(
            'MAELoss|Midas|Optuna|DK2|GRUtanh|Naive',
            case=False, na=False))
    ].copy()
    
    # Fixed cap for cross-plot consistency
    CAP = 50
    
    # Generate one plot per (model, target) combination
    for model_name in tested_models:
        for target_type in ['Price', 'Delta']:
            df_mae_mt = df_mae[
                (df_mae['Model'] == model_name) &
                (df_mae['Target_Type'] == target_type)
            ]
            df_mse_mt = df_mse[
                (df_mse['Model'] == model_name) &
                (df_mse['Target_Type'] == target_type)
            ]
            
            if df_mae_mt.empty or df_mse_mt.empty:
                continue
            
            # Use horizons where MAE loss data exists (matches the experiment design)
            horizons = sorted(df_mae_mt['Horizon'].unique())
            
            if not horizons:
                continue
            
            fig, ax = plt.subplots(figsize=(16, 7))
            
            # Apply cap FIRST before placing labels
            ax.set_ylim(top=CAP)
            
            x_positions = np.arange(len(horizons))
            bar_width = 0.13
            
            # Six bars per horizon: MSE Best/Mean/Worst, MAE Best/Mean/Worst
            configs = [
                ('MSE Best',  '#fdba74', -2.5),
                ('MSE Mean',  '#fb923c', -1.5),
                ('MSE Worst', '#ea580c', -0.5),
                ('MAE Best',  '#93c5fd',  0.5),
                ('MAE Mean',  '#3b82f6',  1.5),
                ('MAE Worst', '#1e3a8a',  2.5),
            ]
            
            for label, color, offset_mult in configs:
                offset = offset_mult * bar_width
                values = []
                for h in horizons:
                    if 'MSE' in label:
                        data = df_mse_mt[df_mse_mt['Horizon'] == h]['MAE'].dropna()
                    else:
                        data = df_mae_mt[df_mae_mt['Horizon'] == h]['MAE'].dropna()
                    
                    if data.empty:
                        values.append(np.nan)
                    elif 'Best' in label:
                        values.append(data.min())
                    elif 'Worst' in label:
                        values.append(data.max())
                    else:
                        values.append(data.mean())
                
                # Clip values for visual display
                display_values = [min(v, CAP) if not np.isnan(v) else np.nan for v in values]
                
                bars = ax.bar(x_positions + offset, display_values, bar_width, 
                             label=label, color=color, alpha=0.85, 
                             edgecolor='white', linewidth=0.5)
                
                # Place value labels appropriately
                for bar, val in zip(bars, values):
                    if np.isnan(val):
                        continue
                    # Values at or near the cap get their label placed INSIDE
                    # the bar (in white) to avoid clipping past the top axis.
                    if val > CAP * 0.95:
                        ax.text(bar.get_x() + bar.get_width()/2, CAP * 0.93,
                               f'{val:.0f}', ha='center', va='top', fontsize=font_bar_value(),
                               color='white', fontweight='bold', rotation=90)
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2, val,
                               f'{val:.1f}', ha='center', va='bottom', 
                               fontsize=font_bar_value(), rotation=90)
            
            # Cap annotation moved to bottom-left so it can't collide with
            # rotated bar labels at the top of the plot.
            ax.annotate(
                f"Y-axis capped at {CAP} for cross-plot consistency. "
                f"Values exceeding cap shown as labels.",
                xy=(0.01, 0.02), xycoords='axes fraction',
                fontsize=font_bar_value(), va='bottom', color='#c0392b',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea', 
                         edgecolor='#c0392b', alpha=0.8)
            )
            
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'{h}h' for h in horizons])
            ax.set_xlabel('Forecast Horizon')
            ax.set_ylabel('MAE Score (Lower is better)')
            ax.set_title(f'Loss Function Alignment: MAE vs MSE Training\n'
                        f'Model: {model_name} | Target: {target_type}')
            # Legend below the chart so it can't collide with bar labels at the top.
            ax.legend(**bottom_legend_kwargs(ncol=6))
            ax.grid(True, axis='y', alpha=0.3)
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Tight layout.*')
                plt.tight_layout()
            
            save_path = os.path.join(
                OUTPUT_DIR, 
                f'Loss_Function_{model_name}_{target_type}.png'
            )
            plt.savefig(save_path, dpi=300)
            print(f"    ✓ {os.path.basename(save_path)}")
            plt.close()


# =====================================================================
# B2. LOSS FUNCTION ALIGNMENT - 24h SUMMARY
# =====================================================================
def plot_loss_function_summary_24h(csv_file):
    """
    One-shot summary of MSE-vs-MAE training at the 24h horizon for each
    NN model. Generates one figure per target, with 3 models on the x-axis
    and 6 grouped bars per model: MSE Best/Mean/Worst, MAE Best/Mean/Worst.
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # Restrict to the three NN models with a configurable loss function.
    # AutoGluon picks its own loss and so is excluded from this comparison.
    NN_MODELS = ['LSTM', 'GRU', 'Transformer']
    
    df_mae = df[df['Experiment'].str.contains('MAELoss', case=False, na=False)].copy()
    if df_mae.empty:
        print("  [SKIP] MAE loss function data not found (summary)")
        return
    
    df_mse = df[
        (df['Version'] == 'Baseline') &
        (df['Model'].isin(NN_MODELS)) &
        (~df['Experiment'].str.contains(
            'MAELoss|Midas|Optuna|DK2|GRUtanh|Naive',
            case=False, na=False))
    ].copy()
    
    # Match the style of the detailed plot so the two read consistently
    CAP = 50
    configs = [
        ('MSE Best',  '#fdba74', -2.5),
        ('MSE Mean',  '#fb923c', -1.5),
        ('MSE Worst', '#ea580c', -0.5),
        ('MAE Best',  '#93c5fd',  0.5),
        ('MAE Mean',  '#3b82f6',  1.5),
        ('MAE Worst', '#1e3a8a',  2.5),
    ]
    bar_width = 0.13
    
    for target_type in ['Price', 'Delta']:
        # Only keep models that have BOTH MSE and MAE 24h data for this target
        models_present = []
        for m in NN_MODELS:
            has_mse = not df_mse[
                (df_mse['Model'] == m) &
                (df_mse['Target_Type'] == target_type) &
                (df_mse['Horizon'] == 24)
            ].empty
            has_mae = not df_mae[
                (df_mae['Model'] == m) &
                (df_mae['Target_Type'] == target_type) &
                (df_mae['Horizon'] == 24)
            ].empty
            if has_mse and has_mae:
                models_present.append(m)
        
        if not models_present:
            print(f"  [SKIP] No paired MSE/MAE 24h data for {target_type}")
            continue
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.set_ylim(top=CAP)
        x_positions = np.arange(len(models_present))
        
        for label, color, offset_mult in configs:
            offset = offset_mult * bar_width
            values = []
            for model in models_present:
                if 'MSE' in label:
                    data = df_mse[
                        (df_mse['Model'] == model) &
                        (df_mse['Target_Type'] == target_type) &
                        (df_mse['Horizon'] == 24)
                    ]['MAE'].dropna()
                else:
                    data = df_mae[
                        (df_mae['Model'] == model) &
                        (df_mae['Target_Type'] == target_type) &
                        (df_mae['Horizon'] == 24)
                    ]['MAE'].dropna()
                
                if data.empty:
                    values.append(np.nan)
                elif 'Best' in label:
                    values.append(data.min())
                elif 'Worst' in label:
                    values.append(data.max())
                else:
                    values.append(data.mean())
            
            display_values = [min(v, CAP) if not np.isnan(v) else np.nan for v in values]
            
            bars = ax.bar(x_positions + offset, display_values, bar_width,
                         label=label, color=color, alpha=0.85,
                         edgecolor='white', linewidth=0.5)
            
            for bar, val in zip(bars, values):
                if np.isnan(val):
                    continue
                # Values at or near the cap get their label placed INSIDE
                # the bar (in white) to avoid clipping past the top axis.
                if val > CAP * 0.95:
                    ax.text(bar.get_x() + bar.get_width() / 2, CAP * 0.93,
                           f'{val:.0f}', ha='center', va='top',
                           fontsize=font_bar_value(),
                           color='white', fontweight='bold', rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2, val,
                           f'{val:.1f}', ha='center', va='bottom',
                           fontsize=font_bar_value(), rotation=90)
        
        # Cap annotation moved to bottom-left so it can't collide with
        # rotated bar labels at the top of the plot.
        ax.annotate(
            f"Y-axis capped at {CAP} for cross-plot consistency. "
            f"Values exceeding cap shown as labels.",
            xy=(0.01, 0.02), xycoords='axes fraction',
            fontsize=font_bar_value(), va='bottom', color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea',
                     edgecolor='#c0392b', alpha=0.8)
        )
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models_present)
        ax.set_xlabel('Neural Network Architecture')
        ax.set_ylabel('MAE Score (Lower is better)')
        ax.set_title(f'Loss Function Alignment: MSE vs MAE Training (24h Summary)\n'
                    f'Target: {target_type}')
        # Legend below the chart so it can't collide with bar labels at the top.
        ax.legend(**bottom_legend_kwargs(ncol=6))
        ax.grid(True, axis='y', alpha=0.3)
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*Tight layout.*')
            plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'Loss_Function_Summary_24h_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# C. HYPERPARAMETER TUNING GAIN (Optuna)
# =====================================================================
def plot_optuna_tuning_gain(csv_file):
    """
    Show improvement from Optuna hyperparameter tuning vs default params.
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # Optuna experiments - prefer new walk-forward Optuna results.
    # Old _Optuna data (broken single-fold objective) is kept as fallback
    # only if the new _OptunaWF rows are not yet present in the CSV.
    df_optuna_wf = df[df['Experiment'].str.contains('OptunaWF', case=False, na=False)].copy()
    if not df_optuna_wf.empty:
        df_optuna = df_optuna_wf
        print("  [INFO] Optuna plot: using walk-forward Optuna data (_OptunaWF)")
    else:
        df_optuna = df[df['Experiment'].str.contains('Optuna', case=False, na=False)].copy()
        if not df_optuna.empty:
            print("  [WARN] Optuna plot: walk-forward data missing, "
                  "falling back to original (broken single-fold) Optuna data.")

    if df_optuna.empty:
        print("  [SKIP] Optuna tuning data not found")
        return

    # Baseline data for comparison. The Version='Baseline' filter excludes
    # all supplementary experiments via plot_utils' updated classifier, but
    # we keep the explicit exclusion regex as a defensive belt-and-braces
    # check in case plot_utils is stale.
    df_default = df[
        (df['Version'] == 'Baseline') &
        (df['Model'].isin(['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest'])) &
        (~df['Experiment'].str.contains(
            'Optuna|Midas|MAELoss|DK2|GRUtanh|Naive',
            case=False, na=False))
    ].copy()
    
    if df_default.empty:
        print("  [SKIP] No default baseline data for comparison")
        return
    
    tree_models = ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest']
    
    for target_type in ['Price', 'Delta']:
        df_opt_t = df_optuna[(df_optuna['Target_Type'] == target_type) & (df_optuna['Horizon'] == 24)]
        df_def_t = df_default[(df_default['Target_Type'] == target_type) & (df_default['Horizon'] == 24)]
        
        if df_opt_t.empty or df_def_t.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(11, 6))
        
        x = np.arange(len(tree_models))
        width = 0.35
        
        default_vals = []
        tuned_vals = []

        # Compare BEST baseline to BEST Optuna-tuned MAE per model.
        # Optuna's purpose is to find the best achievable hyperparameter
        # configuration, so the meaningful comparison is "lowest MAE
        # achievable without tuning" vs "lowest MAE achievable with
        # tuning". Using mean would dilute the comparison: the baseline
        # mean would include all 13 feature sets while the Optuna mean
        # only includes the 3 tuned feature sets (best/mean/worst), so
        # the populations being averaged are not comparable.
        for model in tree_models:
            default = df_def_t[df_def_t['Model'] == model]['MAE'].min()
            tuned = df_opt_t[df_opt_t['Model'] == model]['MAE'].min()
            default_vals.append(default if not np.isnan(default) else 0)
            tuned_vals.append(tuned if not np.isnan(tuned) else 0)
        
        ax.bar(x - width/2, default_vals, width, label='Default Hyperparameters',
               color='#95a5a6', alpha=0.8)
        ax.bar(x + width/2, tuned_vals, width, label='Optuna Tuned',
               color='#9b59b6', alpha=0.8)
        
        # Extend the y-axis ceiling to leave room for the percentage labels
        # above each pair of bars. Without this, the labels clip off the top.
        max_bar = max(max(default_vals), max(tuned_vals))
        ax.set_ylim(top=max_bar * 1.18)
        
        for i, (default, tuned) in enumerate(zip(default_vals, tuned_vals)):
            if default > 0 and tuned > 0:
                ax.text(i - width/2, default, f'{default:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
                ax.text(i + width/2, tuned, f'{tuned:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
                improvement = ((default - tuned) / default) * 100
                color = 'green' if improvement > 0 else 'red'
                ax.text(i, max(default, tuned) + (max_bar * 0.09),
                       f'{"↓" if improvement > 0 else "↑"}{abs(improvement):.1f}%',
                       ha='center', va='center', fontsize=font_bar_value(), color=color, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(tree_models)
        ax.set_xlabel('Tree Model')
        ax.set_ylabel('MAE (Lower is better)')
        ax.set_title(f'Hyperparameter Tuning Impact (Optuna, 30 trials)\n'
                    f'Target: {target_type} | Horizon: 24h')
        ax.legend(**bottom_legend_kwargs(ncol=2))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'Optuna_Tuning_Gain_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# D. AUTOGLUON QUALITY TIERS
# =====================================================================
def plot_autogluon_quality_comparison(csv_file):
    """
    Compare AutoGluon with different quality settings.
    Compares baseline AutoGluon (quick) vs medium_quality (thorough).
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # Medium/best quality experiments (from Stage 3)
    df_quality = df[df['Experiment'].str.contains('AutoGluon_medium_quality|AutoGluon_best_quality', 
                                                   case=False, na=False)].copy()
    
    # Baseline AutoGluon (default quality, quick runs)
    df_baseline = df[
        (df['Model'] == 'AutoGluon') &
        (df['Version'] == 'Baseline') &
        (~df['Experiment'].str.contains(
            'medium_quality|best_quality|Midas|Optuna|MAELoss|DK2|GRUtanh|Naive',
            case=False, na=False))
    ].copy()
    
    if df_quality.empty:
        print("  [SKIP] AutoGluon quality comparison data not found")
        return
    
    # Determine which quality was tested (medium or best)
    quality_label = 'medium_quality' if 'medium_quality' in df_quality['Experiment'].iloc[0].lower() else 'best_quality'
    
    for target_type in ['Price', 'Delta']:
        df_qual_t = df_quality[(df_quality['Target_Type'] == target_type) & (df_quality['Horizon'] == 24)]
        df_base_t = df_baseline[(df_baseline['Target_Type'] == target_type) & (df_baseline['Horizon'] == 24)]
        
        if df_qual_t.empty or df_base_t.empty:
            continue
        
        baseline_mae = df_base_t['MAE'].mean()
        quality_mae = df_qual_t['MAE'].mean()
        baseline_time = df_base_t['Train_Time_Sec'].mean()
        quality_time = df_qual_t['Train_Time_Sec'].mean()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # MAE comparison
        qualities = ['Default\n(quick)', quality_label.replace('_', ' ').title()]
        maes = [baseline_mae, quality_mae]
        colors = ['#3498db', '#e74c3c']
        
        bars1 = ax1.bar(qualities, maes, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        for bar, mae in zip(bars1, maes):
            ax1.text(bar.get_x() + bar.get_width()/2, mae,
                    f'{mae:.2f}', ha='center', va='bottom', fontsize=font_bar_value(), fontweight='bold')
        
        # Extend the y-axis headroom so the percentage label sits clear of both
        # the bar tops and the (two-line) subplot title. We anchor the label
        # at a fraction of the data range rather than relative to bar height,
        # so the position is independent of how tall the bars happen to be.
        ax1.set_ylim(top=max(maes) * 1.18)
        improvement = ((baseline_mae - quality_mae) / baseline_mae) * 100
        color = 'green' if improvement > 0 else 'red'
        symbol = '↓' if improvement > 0 else '↑'
        ax1.text(0.5, max(maes) * 1.09, f'{symbol}{abs(improvement):.1f}%',
                ha='center', va='center', fontsize=font_bar_value(), color=color,
                fontweight='bold', transform=ax1.transData)
        
        ax1.set_ylabel('MAE (Lower is better)')
        ax1.set_title(f'AutoGluon Quality Setting: MAE\nTarget: {target_type}', pad=15)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Training time comparison
        times = [baseline_time/60, quality_time/60]  # Convert to minutes
        bars2 = ax2.bar(qualities, times, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        for bar, t in zip(bars2, times):
            if t < 60:
                label = f'{t:.1f}m'
            else:
                label = f'{t/60:.1f}h'
            ax2.text(bar.get_x() + bar.get_width()/2, t,
                    label, ha='center', va='bottom', fontsize=font_bar_value(), fontweight='bold')
        
        # Same headroom treatment as the left subplot
        ax2.set_ylim(top=max(times) * 1.18)
        time_ratio = quality_time / baseline_time if baseline_time > 0 else 0
        ax2.text(0.5, max(times) * 1.09, f'{time_ratio:.0f}× slower',
                ha='center', va='center', fontsize=font_bar_value(), color='orange',
                fontweight='bold', transform=ax2.transData)
        
        ax2.set_ylabel('Training Time (minutes)')
        ax2.set_title('Training Time Cost', pad=15)
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, f'AutoGluon_Quality_Tiers_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# E. WEATHER DATA SOURCE COMPARISON (DMI vs Midas)
# =====================================================================
def plot_weather_source_comparison(csv_file):
    """
    Compare DMI weather data vs Midas Energy weather data.
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    # Old Midas plot shows the original methodology where Midas data was
    # ADDED on top of the existing feature set. The new MidasSub and
    # MidasRangeDMI experiments use a different methodology (substitution)
    # and have their own plot (plot_midas_substitution_comparison below).
    # Restrict this plot to the original add-on rows only.
    df_midas = df[
        df['Experiment'].str.contains('Midas', case=False, na=False) &
        ~df['Experiment'].str.contains('MidasSub|MidasRange', case=False, na=False)
    ].copy()
    df_dmi = df[
        (df['Version'] == 'Baseline') &
        (~df['Experiment'].str.contains(
            'Midas|Optuna|MAELoss|DK2|GRUtanh|Naive',
            case=False, na=False))
    ].copy()
    
    if df_midas.empty:
        print("  [SKIP] Midas weather data results not found")
        return
    
    models = ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest', 'LSTM', 'GRU', 'Transformer', 'AutoGluon']
    
    for target_type in ['Price', 'Delta']:
        df_midas_t = df_midas[(df_midas['Target_Type'] == target_type) & (df_midas['Horizon'] == 24)]
        df_dmi_t = df_dmi[(df_dmi['Target_Type'] == target_type) & (df_dmi['Horizon'] == 24)]
        
        if df_midas_t.empty or df_dmi_t.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(13, 7))
        
        # Apply cap of 50 for cross-plot consistency
        WEATHER_CAP = 50
        ax.set_ylim(top=WEATHER_CAP)
        
        x = np.arange(len(models))
        width = 0.35
        
        dmi_vals = []
        midas_vals = []
        
        for model in models:
            dmi_mae = df_dmi_t[df_dmi_t['Model'] == model]['MAE'].mean()
            midas_mae = df_midas_t[df_midas_t['Model'] == model]['MAE'].mean()
            dmi_vals.append(dmi_mae if not np.isnan(dmi_mae) else 0)
            midas_vals.append(midas_mae if not np.isnan(midas_mae) else 0)
        
        # Plot bars clipped to cap
        dmi_display = [min(v, WEATHER_CAP) for v in dmi_vals]
        midas_display = [min(v, WEATHER_CAP) for v in midas_vals]
        
        bars_dmi = ax.bar(x - width/2, dmi_display, width, label='DMI Weather Data',
               color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
        bars_midas = ax.bar(x + width/2, midas_display, width, label='Midas Energy Weather Data',
               color='#e67e22', alpha=0.85, edgecolor='white', linewidth=0.5)
        
        # Add labels - either above bar or inside cap area if capped
        for i, (dmi, midas) in enumerate(zip(dmi_vals, midas_vals)):
            if dmi > 0:
                if dmi > WEATHER_CAP:
                    ax.text(i - width/2, WEATHER_CAP * 0.93, f'{dmi:.0f}', 
                           ha='center', va='top', fontsize=font_bar_value(), 
                           color='white', fontweight='bold')
                else:
                    ax.text(i - width/2, dmi, f'{dmi:.1f}', 
                           ha='center', va='bottom', fontsize=font_bar_value())
            
            if midas > 0:
                if midas > WEATHER_CAP:
                    ax.text(i + width/2, WEATHER_CAP * 0.93, f'{midas:.0f}', 
                           ha='center', va='top', fontsize=font_bar_value(),
                           color='white', fontweight='bold')
                else:
                    ax.text(i + width/2, midas, f'{midas:.1f}', 
                           ha='center', va='bottom', fontsize=font_bar_value())
        
        # Cap annotation at the bottom-left for consistency.
        ax.annotate(
            f"Y-axis capped at {WEATHER_CAP} for cross-plot consistency. "
            f"Values exceeding cap shown as labels.",
            xy=(0.01, 0.02), xycoords='axes fraction',
            fontsize=font_bar_value(), va='bottom', color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea', 
                     edgecolor='#c0392b', alpha=0.8)
        )
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        ax.set_xlabel('Model Architecture')
        ax.set_ylabel('MAE (Lower is better)')
        ax.set_title(f'Weather Data Source Comparison: DMI vs Midas Energy\n'
                    f'Target: {target_type} | Horizon: 24h')
        ax.legend(**bottom_legend_kwargs(ncol=2))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'Weather_Source_DMI_vs_Midas_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# F. DK2 GENERALIZATION TEST
# =====================================================================
# Models shown in the DK2 generalization plot. Supervisor requested this be
# restricted to CatBoost (best tree) and Transformer (best NN). To change
# the model selection, edit this list.
DK2_PLOT_MODELS = ['CatBoost', 'Transformer']


def plot_dk2_generalization(csv_file):
    """
    Test if models trained on DK1 generalize to DK2 region.
    Only includes models listed in DK2_PLOT_MODELS.
    """
    df = load_csv(csv_file)
    if df is None:
        return
    
    df_dk2 = df[df['Experiment'].str.contains('DK2', case=False, na=False)].copy()
    df_dk1 = df[
        (df['Version'] == 'Baseline') &
        (df['Region'] == 'DK1') &
        (df['Horizon'] == 24) &
        (~df['Experiment'].str.contains(
            'DK2|Midas|Optuna|MAELoss|GRUtanh|Naive',
            case=False, na=False))
    ].copy()
    
    if df_dk2.empty:
        print("  [SKIP] DK2 generalization data not found")
        return
    
    # Filter to only the models we want to display
    df_dk2 = df_dk2[df_dk2['Model'].isin(DK2_PLOT_MODELS)]
    
    if df_dk2.empty:
        print(f"  [SKIP] No DK2 data for requested models {DK2_PLOT_MODELS}")
        return
    
    # Get models tested on DK2 that are also in our display list
    dk2_models = df_dk2['Model'].unique()
    
    for target_type in ['Price', 'Delta']:
        df_dk2_t = df_dk2[(df_dk2['Target_Type'] == target_type) & (df_dk2['Horizon'] == 24)]
        df_dk1_t = df_dk1[(df_dk1['Target_Type'] == target_type) & (df_dk1['Horizon'] == 24)]
        
        if df_dk2_t.empty or df_dk1_t.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Preserve the order from DK2_PLOT_MODELS rather than alphabetical,
        # so the supervisor's preferred ordering is reflected.
        models = [m for m in DK2_PLOT_MODELS
                  if m in dk2_models and m in df_dk1_t['Model'].values]
        x = np.arange(len(models))
        width = 0.35
        
        dk1_vals = []
        dk2_vals = []
        
        for model in models:
            dk1_mae = df_dk1_t[df_dk1_t['Model'] == model]['MAE'].mean()
            dk2_mae = df_dk2_t[df_dk2_t['Model'] == model]['MAE'].mean()
            dk1_vals.append(dk1_mae if not np.isnan(dk1_mae) else 0)
            dk2_vals.append(dk2_mae if not np.isnan(dk2_mae) else 0)
        
        ax.bar(x - width/2, dk1_vals, width, label='DK1 (Train & Test)',
               color='#2ecc71', alpha=0.8)
        ax.bar(x + width/2, dk2_vals, width, label='DK2 (Test Only)',
               color='#e74c3c', alpha=0.8)
        
        # Extend the y-axis ceiling to leave room for the percentage labels
        # above each pair of bars.
        max_bar = max(max(dk1_vals), max(dk2_vals))
        ax.set_ylim(top=max_bar * 1.18)
        
        for i, (dk1, dk2) in enumerate(zip(dk1_vals, dk2_vals)):
            if dk1 > 0 and dk2 > 0:
                ax.text(i - width/2, dk1, f'{dk1:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
                ax.text(i + width/2, dk2, f'{dk2:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
                degradation = ((dk2 - dk1) / dk1) * 100
                color = 'red' if degradation > 0 else 'green'
                ax.text(i, max(dk1, dk2) + (max_bar * 0.09),
                       f'{"↑" if degradation > 0 else "↓"}{abs(degradation):.0f}%',
                       ha='center', va='center', fontsize=font_bar_value(), color=color, fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE (Lower is better)')
        ax.set_title(f'Cross-Region Generalization: DK1 → DK2\n'
                    f'Target: {target_type} | Horizon: 24h')
        ax.legend(**bottom_legend_kwargs(ncol=2))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f'DK2_Generalization_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# H. GRU ACTIVATION COMPARISON (relu vs tanh)
# =====================================================================
def plot_gru_activation_comparison(csv_file):
    """
    Compares the original GRU (ReLU activation on the recurrent layer)
    against the tanh-activation variant. Shows the best MAE per horizon
    for each variant on the Price target. The tanh variant is expected
    to remove or reduce the spike at horizons 0h, 48h, 72h, 96h that
    motivated the supplementary experiment.

    Looks for rows where:
      - Original GRU: Model=='GRU' and Version=='Baseline'
      - Tanh variant: Model contains 'GRU_tanh' (case-insensitive) OR
                      Experiment contains 'GRUtanh' (case-insensitive)
    """
    df = load_csv(csv_file)
    if df is None:
        return

    # Tanh variant identification - check both Model and Experiment columns
    # because the orchestrator names the model "GRU_tanh" but the experiment
    # name carries the "_GRUtanh" suffix. Either should be enough.
    is_tanh = (
        df['Model'].astype(str).str.contains('GRU_tanh', case=False, na=False) |
        df['Experiment'].str.contains('GRUtanh', case=False, na=False)
    )
    df_tanh = df[is_tanh].copy()

    # Original GRU: must be the baseline GRU runs, not tanh
    df_relu = df[
        (df['Model'] == 'GRU') &
        (df['Version'] == 'Baseline') &
        (~is_tanh)
    ].copy()

    if df_tanh.empty:
        print("  [SKIP] GRU-tanh data not found")
        return
    if df_relu.empty:
        print("  [SKIP] Original (ReLU) GRU baseline data not found")
        return

    # The supplementary experiment was Price-only; we still loop so that
    # if Delta data also exists later it would render automatically.
    targets_present = sorted(set(df_tanh['Target_Type']) & set(df_relu['Target_Type']))
    if not targets_present:
        print("  [SKIP] No overlapping targets between GRU-relu and GRU-tanh")
        return

    for target_type in targets_present:
        df_t_tanh = df_tanh[df_tanh['Target_Type'] == target_type]
        df_t_relu = df_relu[df_relu['Target_Type'] == target_type]

        # Use the union of horizons present in both subsets
        horizons = sorted(set(df_t_tanh['Horizon']) & set(df_t_relu['Horizon']))
        if not horizons:
            print(f"  [SKIP] {target_type}: no overlapping horizons.")
            continue

        # Per the paper's headline plot methodology, take the minimum MAE
        # across feature sets at each horizon for each variant. This gives
        # the best-case comparison and matches Figure 1 in the paper.
        relu_vals = [df_t_relu[df_t_relu['Horizon'] == h]['MAE'].min() for h in horizons]
        tanh_vals = [df_t_tanh[df_t_tanh['Horizon'] == h]['MAE'].min() for h in horizons]

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(horizons))
        width = 0.38

        ax.bar(x - width/2, relu_vals, width, label='GRU (ReLU, default)',
               color='#ff7f0e', alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, tanh_vals, width, label='GRU (tanh)',
               color='#1f77b4', alpha=0.85, edgecolor='white', linewidth=0.5)

        # Headroom for percentage-change labels
        max_bar = max(max(relu_vals), max(tanh_vals))
        ax.set_ylim(top=max_bar * 1.18)

        for i, (r, t) in enumerate(zip(relu_vals, tanh_vals)):
            if np.isnan(r) or np.isnan(t):
                continue
            ax.text(i - width/2, r, f'{r:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
            ax.text(i + width/2, t, f'{t:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
            # Improvement = positive when tanh is better (lower MAE)
            improvement = ((r - t) / r) * 100 if r > 0 else 0
            color = 'green' if improvement > 0 else 'red'
            symbol = '↓' if improvement > 0 else '↑'
            ax.text(i, max(r, t) + (max_bar * 0.09),
                   f'{symbol}{abs(improvement):.0f}%',
                   ha='center', va='center', fontsize=font_bar_value(),
                   color=color, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{h}h' for h in horizons])
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Best MAE per Horizon (Lower is better)')
        ax.set_title(f'GRU Activation Function Comparison\n'
                    f'Target: {target_type}')
        ax.legend(**bottom_legend_kwargs(ncol=2))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f'GRU_Activation_Comparison_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# I. MIDAS SUBSTITUTION (substituted weather vs date-matched DMI control)
# =====================================================================
def plot_midas_substitution_comparison(csv_file):
    """
    Compares the Midas-substitution experiments (where DMI weather was
    replaced with Midas weather) against the date-range-matched DMI
    control. Both run on the same time period (2021-2026), so this is
    a fair apples-to-apples comparison answering: "given the same date
    range, does Midas weather as the only weather source perform
    better, worse, or comparably to DMI weather as the only weather
    source?"

    Looks for rows where:
      - Midas substitution:    Experiment contains 'MidasSub'
      - DMI date-matched ctrl: Experiment contains 'MidasRangeDMI'

    These are the new naming tags introduced by Stage 3 of the
    targeted follow-up orchestrator. The original Midas add-on plot
    (plot_weather_source_comparison above) is unaffected by this
    function - they answer different questions.
    """
    df = load_csv(csv_file)
    if df is None:
        return

    df_midas = df[df['Experiment'].str.contains('MidasSub', case=False, na=False)].copy()
    df_dmi = df[df['Experiment'].str.contains('MidasRangeDMI', case=False, na=False)].copy()

    if df_midas.empty:
        print("  [SKIP] Midas substitution data not found")
        return
    if df_dmi.empty:
        print("  [SKIP] Midas date-matched DMI control data not found")
        return

    # The Midas substitution experiment was 24h horizon only.
    for target_type in ['Price', 'Delta']:
        df_m_t = df_midas[(df_midas['Target_Type'] == target_type) & (df_midas['Horizon'] == 24)]
        df_d_t = df_dmi[(df_dmi['Target_Type'] == target_type) & (df_dmi['Horizon'] == 24)]

        if df_m_t.empty or df_d_t.empty:
            print(f"  [SKIP] {target_type}: missing data for one side of the comparison.")
            continue

        # Take the intersection of models that have BOTH a substitution
        # result and a date-matched control. Anything else cannot be
        # compared fairly and is excluded.
        models = sorted(set(df_m_t['Model']) & set(df_d_t['Model']))
        if not models:
            print(f"  [SKIP] {target_type}: no overlapping models between Midas and DMI control.")
            continue

        midas_vals = []
        dmi_vals = []
        for m in models:
            midas_mae = df_m_t[df_m_t['Model'] == m]['MAE'].mean()
            dmi_mae = df_d_t[df_d_t['Model'] == m]['MAE'].mean()
            midas_vals.append(midas_mae if not np.isnan(midas_mae) else 0)
            dmi_vals.append(dmi_mae if not np.isnan(dmi_mae) else 0)

        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(models))
        width = 0.38

        ax.bar(x - width/2, dmi_vals, width,
               label='DMI weather (date-matched control)',
               color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, midas_vals, width,
               label='Midas weather (substituted)',
               color='#f39c12', alpha=0.85, edgecolor='white', linewidth=0.5)

        # Headroom for percentage labels above each pair
        max_bar = max(max(dmi_vals), max(midas_vals))
        ax.set_ylim(top=max_bar * 1.18)

        for i, (d, m) in enumerate(zip(dmi_vals, midas_vals)):
            if d > 0 and m > 0:
                ax.text(i - width/2, d, f'{d:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
                ax.text(i + width/2, m, f'{m:.1f}', ha='center', va='bottom', fontsize=font_bar_value())
                # Sign convention: positive means Midas is worse than DMI
                degradation = ((m - d) / d) * 100
                color = 'red' if degradation > 0 else 'green'
                symbol = '↑' if degradation > 0 else '↓'
                ax.text(i, max(d, m) + (max_bar * 0.09),
                       f'{symbol}{abs(degradation):.0f}%',
                       ha='center', va='center', fontsize=font_bar_value(),
                       color=color, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0)
        ax.set_xlabel('Model Architecture')
        ax.set_ylabel('MAE (Lower is better)')
        ax.set_title(f'Midas Substitution vs DMI Control (Same Date Range)\n'
                    f'Target: {target_type} | Horizon: 24h')
        ax.legend(**bottom_legend_kwargs(ncol=2))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, f'Midas_Substitution_{target_type}.png')
        plt.savefig(save_path, dpi=300)
        print(f"    {os.path.basename(save_path)}")
        plt.close()


# =====================================================================
# GENERATE ALL SUPPLEMENTARY PLOTS
# =====================================================================
def generate_all_plots(csv_file):
    """Run all supplementary experiment visualizations."""
    print("  Generating supplementary experiment plots...")
    
    plot_naive_baseline_comparison(csv_file)
    plot_loss_function_comparison(csv_file)
    plot_loss_function_summary_24h(csv_file)
    plot_optuna_tuning_gain(csv_file)
    plot_autogluon_quality_comparison(csv_file)
    plot_weather_source_comparison(csv_file)
    plot_dk2_generalization(csv_file)
    plot_gru_activation_comparison(csv_file)
    plot_midas_substitution_comparison(csv_file)


# For standalone testing
if __name__ == "__main__":
    generate_all_plots("../ML_Pipeline/experiment_results_clean.csv")
