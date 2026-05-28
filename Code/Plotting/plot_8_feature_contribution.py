"""
PLOT 8: FEATURE CONTRIBUTION
=============================
Shows progressive MAE improvement as feature groups are added.
Demonstrates the marginal value of each data category.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import load_csv

OUTPUT_DIR = "Plot_8_Feature_Contribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXP_ORDER = [
    ("Exp1", "Weather\nOnly"),
    ("Exp2", "Weather +\nWeatherLags"),
    ("Exp3", "Weather +\nPrices"),
    ("Exp4", "Weather +\nWeatherLags +\nPrices"),
    ("Exp5", "Weather +\nGrid"),
    ("Exp6", "Weather +\nWeatherLags +\nGrid"),
    ("Exp7", "Weather +\nGrid +\nPrices"),
    ("Exp8", "Weather +\nWeatherLags +\nGrid +\nPrices"),
    ("Exp9", "Weather +\nGrid +\nGridLags"),
    ("Exp10", "Weather +\nWeatherLags +\nGrid +\nGridLags"),
    ("Exp11", "Weather +\nGrid +\nGridLags +\nPrices"),
    ("Exp12", "Weather +\nWeatherLags +\nGrid +\nGridLags +\nPrices"),
    ("Exp13", "All\nFeatures")
]


def plot_feature_contribution(csv_file, metric="MAE"):
    df = load_csv(csv_file)
    if df is None:
        return
    
    df = df[df['Version'] == 'Baseline'].copy()
    
    for target_type in ['Price', 'Delta']:
        df_t = df[df['Target_Type'] == target_type]
        horizons = [0, 24, 48, 72] if target_type == 'Price' else [24, 48, 72]
        
        fig, ax = plt.subplots(figsize=(16, 8))
        colors = {0: 'blue', 24: 'green', 48: 'orange', 72: 'red'}
        
        for horizon in horizons:
            df_h = df_t[df_t['Horizon'] == horizon]
            values = []
            for exp_prefix, _ in EXP_ORDER:
                exp_data = df_h[df_h['Base_Experiment'].str.startswith(exp_prefix)]
                values.append(exp_data[metric].mean() if not exp_data.empty else np.nan)
            ax.plot(range(len(EXP_ORDER)), values, marker='o', 
                   label=f'{horizon}h', color=colors.get(horizon, 'grey'), linewidth=2)
        
        ax.set_xticks(range(len(EXP_ORDER)))
        ax.set_xticklabels([label for _, label in EXP_ORDER], rotation=0, ha='center', fontsize=9)
        ax.set_xlabel("Progressive Feature Groups", fontsize=12)
        ax.set_ylabel(f"Mean {metric} (Lower is better)", fontsize=12)
        ax.set_title(f"Feature Contribution Analysis\nTarget: {target_type}", fontsize=14)
        ax.legend(title="Horizon", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(OUTPUT_DIR, f"Feature_Contribution_{target_type}_{metric}.png")
        plt.savefig(save_path, dpi=300)
        print(f"    ✓ {os.path.basename(save_path)}")
        plt.close()


def generate_all_plots(csv_file):
    plot_feature_contribution(csv_file)


if __name__ == "__main__":
    generate_all_plots("../experiment_results_clean.csv")
