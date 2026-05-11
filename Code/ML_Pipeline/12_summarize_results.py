import pandas as pd
import numpy as np

def create_summary_report(input_csv="experiment_results.csv", output_txt="experiment_summary_report.txt"):
    print("Loading data...")
    try:
        df = pd.read_csv(input_csv, sep=None, engine='python')
    except Exception as e:
        print("Failed to load CSV: " + str(e))
        return
        
    if 'Status' in df.columns:
        df = df[df['Status'] == 'SUCCESS'].copy()
        
    print("Processing data...")
    
    df['Target_Type'] = df['Target'].astype(str).apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    df['Horizon'] = df['Target'].astype(str).apply(lambda x: int(x.split('_')[2].replace('h', '')) if len(x.split('_')) > 2 else -1)
    
    def clean_exp_name(name):
        parts = str(name).split('_')
        if len(parts) >= 3 and (parts[-1] in ['Price', 'Delta'] and parts[-2].endswith('h')):
            return "_".join(parts[:-2])
        return name
        
    df['Base_Experiment'] = df['Experiment'].apply(clean_exp_name)
    
    metrics = ['MAE', 'RMSE', 'WMAPE', 'MDA', 'R2', 'sMAPE']
    
    print("Generating report...")
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("=================================================================\n")
        f.write("                 EXPERIMENT SUMMARY REPORT\n")
        f.write("=================================================================\n\n")
        
        # --- NEW DISCLAIMER BLOCK ---
        f.write("METHODOLOGICAL NOTE ON METRICS AND DELTA TARGETS:\n")
        f.write("When predicting 'Delta' (the hour-to-hour change in price), percentage-based\n")
        f.write("metrics like WMAPE mathematically explode. Because the actual target values\n")
        f.write("are clustered near zero, dividing any error by a near-zero denominator\n")
        f.write("creates an artificially massive percentage.\n\n")
        f.write("To ensure a true 1:1 scientific comparison between Absolute Price targets\n")
        f.write("and Delta targets, this summary ranks all experiments using Mean Absolute\n")
        f.write("Error (MAE). MAE represents the average error in raw Euros, stripping away\n")
        f.write("percentage distortions and revealing the true predictive accuracy.\n")
        f.write("-----------------------------------------------------------------\n\n")
        
        # --- PART 1: OVERALL BEST/WORST EXPERIMENTS (NOW USING MAE) ---
        f.write("PART 1: BEST & WORST FEATURE SETS PER HORIZON\n")
        f.write("(Ranked by Average MAE [Raw Euro Error] across all tested models)\n")
        f.write("-----------------------------------------------------------------\n")
        
        for target in sorted(df['Target_Type'].unique()):
            f.write(f"\n[ TARGET TYPE: {target} ]\n")
            df_t = df[df['Target_Type'] == target]
            
            for horizon in sorted(df_t['Horizon'].unique()):
                df_h = df_t[df_t['Horizon'] == horizon]
                
                # Group by base experiment and calculate mean MAE
                exp_scores = df_h.groupby('Base_Experiment')['MAE'].mean().reset_index()
                exp_scores = exp_scores.sort_values('MAE', ascending=True)
                
                if not exp_scores.empty:
                    best = exp_scores.iloc[0]
                    worst = exp_scores.iloc[-1]
                    
                    f.write(f"  Horizon {horizon}h:\n")
                    f.write(f"    -> BEST OVERALL:  {best['Base_Experiment']} (Avg MAE: {best['MAE']:.2f} Euros)\n")
                    f.write(f"    -> WORST OVERALL: {worst['Base_Experiment']} (Avg MAE: {worst['MAE']:.2f} Euros)\n")
                    
        f.write("\n\n")
        
        # --- PART 2: DETAILED METRICS ---
        f.write("PART 2: METRIC BREAKDOWN BY MODEL AND HORIZON\n")
        f.write("-----------------------------------------------------------------\n")
        
        for target in sorted(df['Target_Type'].unique()):
            df_t = df[df['Target_Type'] == target]
            
            for horizon in sorted(df_t['Horizon'].unique()):
                f.write(f"\n==================================================\n")
                f.write(f" HORIZON: {horizon}h  |  TARGET: {target}\n")
                f.write(f"==================================================\n")
                
                df_h = df_t[df_t['Horizon'] == horizon]
                
                for model in sorted(df_h['Model'].unique()):
                    df_m = df_h[df_h['Model'] == model]
                    
                    f.write(f"\n  MODEL: {model}\n")
                    f.write("  " + "-"*40 + "\n")
                    
                    for m in metrics:
                        if m not in df_m.columns:
                            continue
                            
                        valid = df_m.dropna(subset=[m])
                        if valid.empty:
                            continue
                            
                        avg_val = valid[m].mean()
                        max_val = valid[m].max()
                        min_val = valid[m].min()
                        
                        max_exp = valid.loc[valid[m].idxmax(), 'Base_Experiment']
                        min_exp = valid.loc[valid[m].idxmin(), 'Base_Experiment']
                        
                        f.write(f"    {m}:\n")
                        f.write(f"      Average: {avg_val:.4f}\n")
                        f.write(f"      Max:     {max_val:.4f}  (from {max_exp})\n")
                        f.write(f"      Min:     {min_val:.4f}  (from {min_exp})\n")

    print("Done! Summary successfully saved to " + output_txt)

if __name__ == "__main__":
    create_summary_report()