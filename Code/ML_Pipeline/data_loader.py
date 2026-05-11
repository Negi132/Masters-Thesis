import pandas as pd
from ML_Pipeline import config
from pathlib import Path
import json
import os

def load_master_data():
    """Loads the Master Matrix by dynamically building the path."""
    current_file = config.ML_DATA_DIR / f"Master_Matrix_{config.REGION}_Horizon{config.HORIZON}h.csv"
    
    if not current_file.exists():
        raise FileNotFoundError(f"Could not find data at {current_file}")
    
    print(f"Loading {config.REGION} Master Matrix for Horizon {config.HORIZON}h...")
    df = pd.read_csv(current_file)
    
    df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True)
    df = df.sort_values('HourUTC').reset_index(drop=True)
    
    return df

def get_filtered_features(df, active_groups=["All_Features"], model_name=None):
    """
    Subsets data using explicit lists and auto-cleans time features for NNs.
    Now includes JSON-based feature pruning integration.
    """
    # 1. Determine if we should drop ordinal time (Auto-detect NN)
    nn_models = ["LSTM", "Transformer", "GRU", "NeuralNet"]
    drop_ordinal_time = any(nn in str(model_name) for nn in nn_models) if model_name else False

    if "All_Features" in active_groups:
        keep_cols = df.columns.tolist()
    else:
        keep_cols = ["HourUTC", config.TARGET_COL] 
        
        for group_name in active_groups:
            if group_name in config.COL_GROUPS:
                keep_cols.extend(config.COL_GROUPS[group_name])
            else:
                print(f"  [WARNING] Group '{group_name}' not found in config.COL_GROUPS")
        
        keep_cols = list(dict.fromkeys(keep_cols))
        
    existing_cols = [c for c in keep_cols if c in df.columns]
    filtered_df = df[existing_cols].copy()

    # 2. AUTO-CLEAN: The NN Time Guard
    if drop_ordinal_time:
        ordinals = ["hour", "month", "dayofweek", "dayofyear", "year"]
        to_remove = [c for c in ordinals if c in filtered_df.columns]
        if to_remove:
            filtered_df = filtered_df.drop(columns=to_remove)
            print(f"  [AUTO-CLEAN] Removed ordinal time features for {model_name}")

    # 3. STANDARD SHIELDS
    current_leaky = config.get_leaky_columns(config.HORIZON)
    to_drop = [c for c in current_leaky if c != config.TARGET_COL]
    to_drop.append('PriceArea')
    
    if config.HORIZON == 0:
        deltas = [c for c in filtered_df.columns if "historical_delta" in c]
        to_drop.extend(deltas)

    final_features = [c for c in filtered_df.columns if c not in to_drop]
    df_pre_prune = filtered_df[final_features].select_dtypes(include=['number', 'datetime64[ns, UTC]'])

    # =========================================================================
    # 4. THE PRUNING ENGINE (JSON Integration)
    # =========================================================================
    if config.USE_PRUNING_ENGINE and model_name:
        try:
            target_type = config.TARGET_COL.split('_')[1] 
            json_file = f"pruned_features_{target_type}.json"
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    pruned_dict = json.load(f)
                        
                base_exp = config.EXPERIMENT_NAME
                for split_str in ['_0h', '_24h', '_48h', '_72h', '_96h', '_120h', '_144h', '_168h']:
                    if split_str in base_exp:
                        base_exp = base_exp.split(split_str)[0]
                        break
                            
                dict_key = f"{model_name}_{base_exp}"
                    
                if dict_key in pruned_dict:
                    allowed_features = pruned_dict[dict_key]
                    keep_cols = [c for c in df_pre_prune.columns if c in allowed_features or c in ["HourUTC", config.TARGET_COL]]
                        
                    dropped_count = len(df_pre_prune.columns) - len(keep_cols)
                    if dropped_count > 0:
                        print(f"  [PRUNING ENGINE] Activated for {dict_key}. Dropped {dropped_count} noisy features.")
                            
                    return df_pre_prune[keep_cols]
                    
        except Exception as e:
            print(f"  [PRUNING WARNING] Could not apply pruning JSON: {e}")
            
    return df_pre_prune

def get_train_test_split(df, train_start_idx, train_end_idx, test_end_idx):
    train_slice = df.iloc[train_start_idx : train_end_idx].copy()
    test_slice = df.iloc[train_end_idx : test_end_idx].copy()
    
    X_train = train_slice.drop(columns=[config.TARGET_COL, 'HourUTC'])
    y_train = train_slice[config.TARGET_COL]
    
    X_test = test_slice.drop(columns=[config.TARGET_COL, 'HourUTC'])
    y_test = test_slice[config.TARGET_COL]
    
    return X_train, y_train, X_test, y_test