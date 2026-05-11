from pathlib import Path

# =============================================================================
# 1. FILE PATHS & EXPERIMENT TRACKING
# =============================================================================
# We go UP from ML_Pipeline, then DOWN into Data_Engineering/Data
from pathlib import Path

# =============================================================================
# 1. FILE PATHS & EXPERIMENT TRACKING
# =============================================================================
BASE_PATH = Path("../Data_Engineering/Data")
ML_DATA_DIR = BASE_PATH / "ML_Ready_Data"
USE_PRUNING_ENGINE = False

# These will be overwritten dynamically by the experiment runner
REGION = "DK1" 
HORIZON = 0
TARGET_COL = "TARGET_Price_0h"
EXPERIMENT_NAME = "Default_Run"
ACTIVE_GROUPS = ["All_Features"]

EXPERIMENT_LOG = "experiment_results.csv"

def get_leaky_columns(horizon):
    """Dynamically generates the leaky columns based on the current horizon."""
    return [
        f'TARGET_Price_{horizon}h', 
        f'TARGET_Delta_{horizon}h',
        'SpotPriceEUR' 
    ]

# =============================================================================
# 2. COLUMN GROUPS
# =============================================================================
COL_GROUPS = {
    "Weather": [
        "acc_precip", "bright_sunshine", "leaf_moisture", "max_temp_w_date", "max_wind_speed_10min", "max_wind_speed_3sec", "mean_pressure", "mean_radiation", "mean_relative_hum", "mean_temp", "mean_wind_dir", "mean_wind_speed", "min_temp", "temp_grass", "temp_soil_10", "vapour_pressure_deficit_mean" 
    ],
    "WeatherLags":  [
        "acc_precip_lag_24h", "bright_sunshine_lag_24h", "leaf_moisture_lag_24h", "max_temp_w_date_lag_24h", "max_wind_speed_10min_lag_24h", "max_wind_speed_3sec_lag_24h", "mean_pressure_lag_24h", "mean_radiation_lag_24h", "mean_relative_hum_lag_24h", "mean_temp_lag_24h", "mean_wind_dir_lag_24h", "mean_wind_speed_lag_24h", "min_temp_lag_24h", "temp_grass_lag_24h", "temp_soil_10_lag_24h", "vapour_pressure_deficit_mean_lag_24h"
    ],
    "Grid": [
        "CentralPowerMWh", "LocalPowerMWh", "CommercialPowerMWh", "LocalPowerSelfConMWh", "OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh", "OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh", "HydroPowerMWh", "SolarPowerLt10kW_MWh", "SolarPowerGe10Lt40kW_MWh", "SolarPowerGe40kW_MWh", "SolarPowerSelfConMWh", "UnknownProdMWh", "GrossConsumptionMWh", "GridLossTransmissionMWh", "GridLossInterconnectorsMWh", "GridLossDistributionMWh", "PowerToHeatMWh", "CommercialPowerMWh_imputed", "UnknownProdMWh_imputed", "GridLossInterconnectorsMWh_imputed", "GridLossDistributionMWh_imputed"
    ],
    "GridLags": [
        "CentralPowerMWh_lag_24h", "LocalPowerMWh_lag_24h", "CommercialPowerMWh_lag_24h", "LocalPowerSelfConMWh_lag_24h", "OffshoreWindLt100MW_MWh_lag_24h", "OffshoreWindGe100MW_MWh_lag_24h", "OnshoreWindLt50kW_MWh_lag_24h", "OnshoreWindGe50kW_MWh_lag_24h", "HydroPowerMWh_lag_24h", "SolarPowerLt10kW_MWh_lag_24h", "SolarPowerGe10Lt40kW_MWh_lag_24h", "SolarPowerGe40kW_MWh_lag_24h", "SolarPowerSelfConMWh_lag_24h", "UnknownProdMWh_lag_24h", "GrossConsumptionMWh_lag_24h", "GridLossTransmissionMWh_lag_24h", "GridLossInterconnectorsMWh_lag_24h", "GridLossDistributionMWh_lag_24h", "PowerToHeatMWh_lag_24h", "CommercialPowerMWh_imputed_lag_24h", "UnknownProdMWh_imputed_lag_24h", "GridLossInterconnectorsMWh_imputed_lag_24h", "GridLossDistributionMWh_imputed_lag_24h"  
    ],
    "GridExchange": [
        "ExchangeNO_MWh", "ExchangeSE_MWh", "ExchangeGE_MWh", "ExchangeNL_MWh", "ExchangeGB_MWh", "ExchangeGreatBelt_MWh"
    ],
    "GridExchangeLags": [
        "ExchangeNO_MWh_lag_24h", "ExchangeSE_MWh_lag_24h", "ExchangeGE_MWh_lag_24h", "ExchangeNL_MWh_lag_24h", "ExchangeGB_MWh_lag_24h", "ExchangeGreatBelt_MWh_lag_24h"
    ],
    "Prices": [
        "SpotPriceEUR_lag_24h", "SpotPriceEUR_lag_48h", "SpotPriceEUR_lag_168h", "DayAheadPriceEUR", "DayAheadPriceEUR_lag_24h", "SpotPriceEUR_historical_delta_24h", "SpotPriceEUR_historical_delta_48h", "SpotPriceEUR_historical_delta_168h"
    ],
    "Time": [
        "hour", "month", "dayofweek", "dayofyear", "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"
    ],
    "Target":   [
        "TARGET_Price_0h", "TARGET_Delta_0h"
    ],
    "All_features":  [
        "ALL"
    ]
}

# =============================================================================
# 3. WALK-FORWARD VALIDATION SETTINGS
# =============================================================================
INITIAL_TRAIN_DAYS = 365 * 2  
TEST_DAYS = 30                
STEP_DAYS = 30                

# =============================================================================
# 4. MODEL TOGGLES
# =============================================================================
RUN_XGBOOST = True
RUN_LIGHTGBM = True
RUN_CATBOOST = True
RUN_RANDOM_FOREST = True
RUN_LSTM = False
RUN_GRU = False
RUN_TRANSFORMER = False
RUN_AUTOGLUON = False