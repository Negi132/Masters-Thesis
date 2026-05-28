"""
Midas Master Matrix Builder
============================
Builds an ML-ready master matrix where the DMI weather columns are
replaced with Midas weather columns. Produces one file per horizon
following the same naming convention as the existing matrices, so
the existing data_loader can find them by setting config.REGION
to a special value.

Output files:
    Data/ML_Ready_Data/Master_Matrix_DK1_Midas_Horizon{H}h.csv

These files keep all non-weather columns from the existing DMI master
matrices (grid, prices, time, targets, leaks) and substitute the
Weather and WeatherLags groups with Midas equivalents.

Run this BEFORE running the orchestrator. The orchestrator can
also call this if the files are missing.
"""

import json
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Allow imports from the ML_Pipeline package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ML_Pipeline import config


# =====================================================================
# CONFIGURATION
# =====================================================================
# Adjust this path if your Midas data lives elsewhere
MIDAS_DATA_DIR = Path("../Data_Engineering/Data/DMI/weather-data")
MIDAS_FILE_DK1 = MIDAS_DATA_DIR / "weather-dk1.json"

# Midas weather columns in the JSON. The script will rename them to
# avoid name collisions with anything else and to make the substitution
# obvious in the CSV output.
MIDAS_RAW_COLUMNS = [
    "avg_temp_dry",
    "avg_humidity",
    "avg_cloud_cover",
    "avg_wind_dir",
    "avg_wind_speed",
    "avg_radia_glob_past1h",
    "avg_sun_last1h_glob",
]

# Renamed columns to match the "midas_*" pattern. This makes filtering
# trivial and keeps the master matrix self-documenting.
MIDAS_RENAME = {col: f"midas_{col[4:]}" if col.startswith("avg_") else f"midas_{col}"
                for col in MIDAS_RAW_COLUMNS}

# Universal lag applied to weather features (matches Script 9)
UNIVERSAL_LAG_HOURS = 24

HORIZONS = [0, 24, 48, 72, 96, 120, 144, 168]


# =====================================================================
# MIDAS GROUP REGISTRATION
# =====================================================================
def get_midas_column_groups():
    """
    Returns the Midas equivalents of the Weather and WeatherLags groups.
    The orchestrator monkey-patches config.COL_GROUPS with these so
    feature filtering works transparently.
    """
    midas_weather = list(MIDAS_RENAME.values())
    midas_weather_lags = [f"{col}_lag_{UNIVERSAL_LAG_HOURS}h" for col in midas_weather]
    return {
        "MidasWeather":     midas_weather,
        "MidasWeatherLags": midas_weather_lags,
    }


# =====================================================================
# LOADING AND PARSING
# =====================================================================
def load_midas_dataframe(json_path=MIDAS_FILE_DK1):
    """Reads the Midas JSON file and returns a tidy hourly DataFrame."""
    print(f"  Loading Midas JSON: {json_path}")
    if not Path(json_path).exists():
        raise FileNotFoundError(f"Midas JSON not found at {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"  Raw Midas rows: {len(df):,}")

    # Parse timestamps to UTC and normalise the column name
    df['HourUTC'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.drop(columns=['datetime'])

    # Floor to the hour to be defensive against any 15-minute artifacts
    df['HourUTC'] = df['HourUTC'].dt.floor('h')

    # Sort and dedupe
    df = df.sort_values('HourUTC').drop_duplicates(subset='HourUTC', keep='last')
    df = df.reset_index(drop=True)

    # Keep only the columns we care about
    keep = ['HourUTC'] + MIDAS_RAW_COLUMNS
    df = df[keep]

    # Rename to the midas_* convention
    df = df.rename(columns=MIDAS_RENAME)

    # Coerce numeric (in case any value imported as string)
    for col in MIDAS_RENAME.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  Midas date range: {df['HourUTC'].min()} -> {df['HourUTC'].max()}")
    print(f"  Midas hourly rows after dedup: {len(df):,}")
    return df


def add_midas_lags(df_midas):
    """Adds 24h universal lag columns for all Midas features."""
    midas_cols = [c for c in df_midas.columns if c != 'HourUTC']
    for col in midas_cols:
        df_midas[f"{col}_lag_{UNIVERSAL_LAG_HOURS}h"] = df_midas[col].shift(UNIVERSAL_LAG_HOURS)
    return df_midas


# =====================================================================
# MASTER MATRIX REBUILD
# =====================================================================
def identify_dmi_weather_columns():
    """
    Returns the union of Weather and WeatherLags column names from
    config.COL_GROUPS. These are the columns we strip from the DMI
    master matrix before splicing in Midas data.
    """
    weather = config.COL_GROUPS.get("Weather", [])
    weather_lags = config.COL_GROUPS.get("WeatherLags", [])
    return set(weather) | set(weather_lags)


def build_midas_matrix_for_horizon(horizon, df_midas_with_lags, force=False):
    """
    Builds Master_Matrix_DK1_Midas_Horizon{H}h.csv by:
      1. Loading the existing DMI matrix for this horizon
      2. Dropping all DMI weather + weather_lag columns
      3. Inner-joining with Midas data on HourUTC
      4. Writing the result to disk
    """
    src_path = config.ML_DATA_DIR / f"Master_Matrix_DK1_Horizon{horizon}h.csv"
    dst_path = config.ML_DATA_DIR / f"Master_Matrix_DK1_Midas_Horizon{horizon}h.csv"

    if dst_path.exists() and not force:
        print(f"  Skipping horizon {horizon}h - already exists at {dst_path.name}")
        return

    if not src_path.exists():
        print(f"  [WARN] Source DMI matrix not found: {src_path} - skipping horizon {horizon}h")
        return

    print(f"\n  Building horizon {horizon}h:")
    df = pd.read_csv(src_path)
    df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True)
    print(f"    Loaded DMI matrix: {len(df):,} rows, {df.shape[1]} cols")

    # Drop DMI weather columns
    dmi_weather_cols = identify_dmi_weather_columns()
    cols_to_drop = [c for c in df.columns if c in dmi_weather_cols]
    df = df.drop(columns=cols_to_drop)
    print(f"    Dropped {len(cols_to_drop)} DMI weather columns")

    # Also drop any "_imputed" weather flags - they belong with DMI weather
    imputed_weather = [c for c in df.columns
                       if c.endswith('_imputed') and any(w in c for w in ['temp', 'wind', 'sun',
                                                                          'humid', 'cloud', 'precip',
                                                                          'pressure', 'radia', 'leaf'])]
    if imputed_weather:
        df = df.drop(columns=imputed_weather)
        print(f"    Dropped {len(imputed_weather)} associated weather _imputed flags")

    # Inner-join on HourUTC: only keep rows where both DMI infrastructure
    # data AND Midas weather data are present
    df_merged = pd.merge(df, df_midas_with_lags, on='HourUTC', how='inner')
    print(f"    After inner-join with Midas: {len(df_merged):,} rows")
    print(f"    Date range: {df_merged['HourUTC'].min()} -> {df_merged['HourUTC'].max()}")

    # Drop any Midas-lag-induced NaN rows at the start of the date range
    lag_cols = [c for c in df_merged.columns if c.endswith(f'_lag_{UNIVERSAL_LAG_HOURS}h')
                and c.startswith('midas_')]
    if lag_cols:
        before = len(df_merged)
        df_merged = df_merged.dropna(subset=lag_cols).reset_index(drop=True)
        print(f"    Dropped {before - len(df_merged):,} rows with NaN Midas lags")

    # Write back with the timestamp string format the existing pipeline uses
    df_merged['HourUTC'] = df_merged['HourUTC'].dt.strftime('%Y-%m-%dT%H:%M:%S+00:00')
    df_merged.to_csv(dst_path, index=False)
    print(f"    Saved: {dst_path.name} ({df_merged.shape[1]} cols)")


def get_midas_date_range(df_midas_with_lags):
    """Returns the (min, max) HourUTC after lag-induced trimming."""
    # Trim leading NaNs from lags
    lag_cols = [c for c in df_midas_with_lags.columns
                if c.endswith(f'_lag_{UNIVERSAL_LAG_HOURS}h')]
    df = df_midas_with_lags.dropna(subset=lag_cols)
    return df['HourUTC'].min(), df['HourUTC'].max()


def main(force=False):
    """Build all Midas master matrices from scratch."""
    print("=" * 70)
    print("  MIDAS MASTER MATRIX BUILDER")
    print("=" * 70)

    # 1. Load and prepare Midas data once
    df_midas = load_midas_dataframe()
    df_midas = add_midas_lags(df_midas)

    # 2. Build matrices for each horizon
    for h in HORIZONS:
        build_midas_matrix_for_horizon(h, df_midas, force=force)

    print("\n" + "=" * 70)
    print("  MIDAS MATRIX BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if output matrices exist")
    args = p.parse_args()
    main(force=args.force)
