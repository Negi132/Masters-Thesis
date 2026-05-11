import pandas as pd
import numpy as np
import time
import pickle
import traceback
from pathlib import Path

# Scikit-Learn / Boosting Imports
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Deep Learning Imports (TensorFlow/Keras)
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization, Add, Flatten
from keras.callbacks import EarlyStopping

# AutoGluon Import
from autogluon.tabular import TabularPredictor

from ML_Pipeline import config
from ML_Pipeline import data_loader
from ML_Pipeline import evaluator


# =============================================================================
# 1. MODEL WRAPPERS (Translators for NNs and AutoGluon)
# =============================================================================

class KerasRNNWrapper:
    """Wrapper for Keras LSTM and GRU models to handle 3D tensors dynamically."""
    def __init__(self, rnn_type='LSTM', epochs=50, batch_size=64):
        self.rnn_type = rnn_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.train_loss = []
        self.val_loss = []

    def fit(self, X, y, validation_data=None, verbose=0):
        # Dynamically build the model on the first fit so it knows the feature count
        if self.model is None:
            self.model = Sequential()
            # X.shape is (samples, timesteps, features)
            input_shape = (X.shape[1], X.shape[2])
            
            if self.rnn_type == 'LSTM':
                self.model.add(LSTM(64, activation='relu', input_shape=input_shape))
            elif self.rnn_type == 'GRU':
                self.model.add(GRU(64, activation='relu', input_shape=input_shape))
                
            self.model.add(Dropout(0.2))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dense(1)) 
            
            self.model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks = [early_stop] if validation_data else []

        history = self.model.fit(
            X, y, 
            epochs=self.epochs, 
            batch_size=self.batch_size,
            validation_data=validation_data, 
            verbose=verbose, 
            callbacks=callbacks
        )
        
        # Save learning curves
        self.train_loss = history.history.get('loss', [])
        self.val_loss = history.history.get('val_loss', [])

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

class KerasTransformerWrapper:
    """Wrapper for a lightweight Time-Series Transformer."""
    def __init__(self, epochs=50, batch_size=64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.train_loss = []
        self.val_loss = []

    def fit(self, X, y, validation_data=None, verbose=0):
        if self.model is None:
            # X.shape is (samples, timesteps, features)
            input_shape = (X.shape[1], X.shape[2])
            inputs = Input(shape=input_shape)
            
            # 1. Multi-Head Attention Block
            attention_out = MultiHeadAttention(key_dim=32, num_heads=2)(inputs, inputs)
            attention_out = Add()([inputs, attention_out]) # Residual connection
            attention_out = LayerNormalization()(attention_out)
            
            # 2. Feed-Forward Block
            # Compress to 32 neurons to find patterns...
            ff_hidden = Dense(32, activation='relu')(attention_out)
            # ...then project back to the original feature size (X.shape[2]) to allow addition
            ff_out = Dense(X.shape[2])(ff_hidden)
            
            ff_out = Add()([attention_out, ff_out]) # Residual connection
            ff_out = LayerNormalization()(ff_out)
            
            # 3. Output Block
            x = Flatten()(ff_out)
            x = Dropout(0.2)(x)
            outputs = Dense(1)(x)
            
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        callbacks = [early_stop] if validation_data else []

        history = self.model.fit(
            X, y, 
            epochs=self.epochs, 
            batch_size=self.batch_size,
            validation_data=validation_data, 
            verbose=verbose, 
            callbacks=callbacks
        )
        
        self.train_loss = history.history.get('loss', [])
        self.val_loss = history.history.get('val_loss', [])

    def predict(self, X):
        return self.model.predict(X, verbose=0).flatten()

class AutoGluonWrapper:
    """Wrapper to handle AutoGluon's unique TabularPredictor logic."""
    def __init__(self, time_limit=6, path="AutogluonModels"):
        self.time_limit = time_limit 
        self.path = path
        self.predictor = None

    def fit(self, train_data, label, **kwargs):
        self.predictor = TabularPredictor(label=label, path=self.path, verbosity=0).fit(
            train_data, 
            time_limit=self.time_limit,
            hyperparameters='very_light', 
            presets='medium_quality' # Note: Change to 'best_quality' for final thesis run
        )

    def predict(self, X_test):
        return self.predictor.predict(X_test).values


# =============================================================================
# 2. MODEL DICTIONARY
# =============================================================================

def get_models():
    """Returns a dictionary of models to be trained based on config toggles."""
    models = {}
    
    # --- TREE MODELS ---
    if getattr(config, 'RUN_XGBOOST', False):
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, n_jobs=-1)
    if getattr(config, 'RUN_LIGHTGBM', False):
        models["LightGBM"] = LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, n_jobs=-1, verbose=-1)
    if getattr(config, 'RUN_CATBOOST', False):
        models["CatBoost"] = CatBoostRegressor(n_estimators=100, learning_rate=0.1, depth=6, verbose=0)
    if getattr(config, 'RUN_RANDOM_FOREST', False):
        models["RandomForest"] = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1)

# --- NEURAL NETWORKS ---
    if getattr(config, 'RUN_LSTM', False):
        models["LSTM"] = KerasRNNWrapper(rnn_type='LSTM', epochs=50, batch_size=64)
    if getattr(config, 'RUN_GRU', False):
        models["GRU"] = KerasRNNWrapper(rnn_type='GRU', epochs=50, batch_size=64)
    if getattr(config, 'RUN_TRANSFORMER', False):
        models["Transformer"] = KerasTransformerWrapper(epochs=50, batch_size=64)

    # --- AUTOGLUON ---
    if getattr(config, 'RUN_AUTOGLUON', False):
        models["AutoGluon"] = AutoGluonWrapper(time_limit=6)

    return models


# =============================================================================
# 3. PIPELINE EXECUTION
# =============================================================================

def run_walk_forward_pipeline():
    """Executes the full walk-forward training loop."""
    raw_df = data_loader.load_master_data()
    
    models_to_run = get_models()
    if not models_to_run:
        print("No models selected in config. Skipping...")
        return

    # Dictionary to store predictions, times, AND learning curves
    all_results = {name: {"y_true": [], "y_pred": [], "times": [], "learning_curves": []} for name in models_to_run}

    for name, model in models_to_run.items():
        print(f"\n--- Starting Pipeline for Model: {name} ---")
        
        # 1. AUTO-FILTER (Handles dropping ordinals for NNs automatically)
        df = data_loader.get_filtered_features(
            raw_df, 
            active_groups=config.ACTIVE_GROUPS, 
            model_name=name
        )

        total_rows = len(df)
        train_size = config.INITIAL_TRAIN_DAYS * 24
        test_size = config.TEST_DAYS * 24
        step_size = config.STEP_DAYS * 24
        
        current_train_end = train_size
        window_count = 0

        print(f"Starting Walk-Forward Validation (Step: {config.STEP_DAYS} days)...")
        
        while current_train_end + test_size <= total_rows:
            window_count += 1
            train_start = 0 # Expanding window
            
            X_train, y_train, X_test, y_test = data_loader.get_train_test_split(
                df, train_start, current_train_end, current_train_end + test_size
            )

            start_t = time.time()
            try:
                # =========================================================
                # FAILSAFE: CONSTANT TARGET CHECK
                # =========================================================
                if y_train.nunique() <= 1:
                    constant_val = y_train.iloc[0] if len(y_train) > 0 else 0
                    preds = np.full(len(X_test), constant_val)
                    
                    # Log empty curves and skip fitting
                    all_results[name]["learning_curves"].append(None)
                else: 
                    # =========================================================
                    # SCENARIO A: NEURAL NETWORKS
                    # =========================================================
                    if any(nn in name for nn in ["LSTM", "GRU", "Transformer"]):
                        scaler = StandardScaler()
                        X_train_proc = scaler.fit_transform(X_train)
                        X_test_proc = scaler.transform(X_test)
                        
                        # Reshape to 3D Tensor: [samples, 1 timestep, features]
                        X_train_proc = X_train_proc.reshape((X_train_proc.shape[0], 1, X_train_proc.shape[1]))
                        X_test_proc = X_test_proc.reshape((X_test_proc.shape[0], 1, X_test_proc.shape[1]))
                        
                        model.fit(X_train_proc, y_train, validation_data=(X_test_proc, y_test), verbose=0)
                        preds = model.predict(X_test_proc)
                        
                        # Extract Learning Curves
                        curves = {'train': model.train_loss, 'val': model.val_loss}
                        all_results[name]["learning_curves"].append(curves)

                    # =========================================================
                    # SCENARIO B: AUTOGLUON
                    # =========================================================
                    elif "AutoGluon" in name:
                        train_data = X_train.copy()
                        train_data[config.TARGET_COL] = y_train.values
                        
                        model.fit(train_data, label=config.TARGET_COL)
                        preds = model.predict(X_test)
                        
                        all_results[name]["learning_curves"].append(None)

                    # =========================================================
                    # SCENARIO C: TREE MODELS
                    # =========================================================
                    else:
                        fit_kwargs = {}
                        if name in ["XGBoost", "CatBoost"]:
                            fit_kwargs['eval_set'] = [(X_train, y_train), (X_test, y_test)]
                            fit_kwargs['verbose'] = False
                        elif name == "LightGBM":
                            fit_kwargs['eval_set'] = [(X_train, y_train), (X_test, y_test)]
                        
                        model.fit(X_train, y_train, **fit_kwargs)
                        preds = model.predict(X_test)
                        
                        # Safely Extract Tree Learning Curves
                        curves = {'train': [], 'val': []}
                        try:
                            if name == "XGBoost":
                                res = model.evals_result()
                                metric = list(res['validation_0'].keys())[0]
                                curves['train'] = res['validation_0'][metric]
                                curves['val'] = res['validation_1'][metric]
                            elif name == "LightGBM":
                                res = model.evals_result_
                                metric = list(res['training'].keys())[0]
                                curves['train'] = res['training'][metric]
                                curves['val'] = res['valid_1'][metric]
                            elif name == "CatBoost":
                                res = model.get_evals_result()
                                metric = list(res['learn'].keys())[0]
                                curves['train'] = res['learn'][metric]
                                curves['val'] = res['validation'][metric]
                            else:
                                curves = None
                        except Exception as e:
                            curves = None # Failsafe if metric extraction fails
                            
                        all_results[name]["learning_curves"].append(curves)

                    elapsed = time.time() - start_t
                    
                    # Save Results
                    all_results[name]["y_true"].extend(y_test.values)
                    all_results[name]["y_pred"].extend(preds)
                    all_results[name]["times"].append(elapsed)
                    
                    if window_count % 10 == 0:
                        print(f"  [{name}] Window {window_count} processed...")

            except Exception as e:
                print(f"  [ERROR] {name} failed in window {window_count}: {e}")
                traceback.print_exc()

            current_train_end += step_size

    # --- Final Evaluation & Logging ---
    print("\nAggregation & Final Evaluation:")
    for name in models_to_run:
        if len(all_results[name]["y_true"]) == 0: continue
        
        metrics = evaluator.calculate_metrics(
            all_results[name]["y_true"],
            all_results[name]["y_pred"],
            np.sum(all_results[name]["times"])
        )
            
        metrics["Model"] = name
        evaluator.log_experiment(metrics)
        print(f"  {name:<12} | WMAPE: {metrics['WMAPE']:.2f}% | MDA: {metrics['MDA']:.2f}%")

    # --- SAVE LEARNING CURVES TO DISK FOR THESIS ---
    results_dir = Path("Experiment_Logs")
    results_dir.mkdir(exist_ok=True)

    file_name = results_dir / f"{config.EXPERIMENT_NAME}_{config.REGION}_{config.HORIZON}h_Results.pkl"

    # Safely merge with existing data rather than overwriting
    if file_name.exists():
        try:
            with open(file_name, 'rb') as f:
                existing_data = pickle.load(f)
            existing_data.update(all_results)
            all_results = existing_data
            print(f"  [MERGE] Merged with existing results in {file_name.name}")
        except Exception as e:
            print(f"  [WARNING] Could not merge with existing pickle, overwriting: {e}")

    with open(file_name, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"  [SAVED] Raw predictions and learning curves saved to {file_name}")