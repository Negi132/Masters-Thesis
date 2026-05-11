"""
PREPROCESSING OVERVIEW AUDIT
==============================
Gives a concise overview of what happened to each data source
during preprocessing, organised by source type:
  - Price data
  - Production & Consumption data
  - Weather data (DK1 and DK2 separately)

For each source reports:
  - Raw rows and columns
  - Columns dropped and why
  - Missing values and how they were filled
  - Final rows and columns fed into model training

Run from the project root directory (same level as the Data folder).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================================
# CONFIGURATION
# =====================================================================
BASE_PATH           = Path("Data")
MISSING_THRESHOLD   = 0.90
DOMINANCE_THRESHOLD = 0.95
INTERPOLATION_LIMIT = 12

DMI_PROCESSED_DIR = BASE_PATH / "DMI" / "ProcessedZones"
DMI_ALIGNED_DIR   = BASE_PATH / "DMI" / "AlignedZones"
ALIGNED_DIR       = BASE_PATH / "Aligned_Yearly"
ML_READY_DIR      = BASE_PATH / "ML_Ready_Data"

ALWAYS_KEEP = ['HourUTC', 'PriceArea', 'SpotPriceEUR', 'DayAheadPriceEUR']
REGIONS     = ["DK1", "DK2"]


# =====================================================================
# HELPERS
# =====================================================================

def pct(num, den):
    if den == 0:
        return "N/A"
    return f"{num / den * 100:.4f}%"

def row(label, value, extra=""):
    extra_str = f"  ({extra})" if extra else ""
    print(f"  {label:<45} {str(value):>12}{extra_str}")

def divider(char="=", width=70):
    print(char * width)

def header(title):
    print()
    divider()
    print(f"  {title}")
    divider()

def load_df(path, filter_dk=False):
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    if filter_dk and 'PriceArea' in df.columns:
        df = df[df['PriceArea'].isin(['DK1', 'DK2'])].copy()
    return df

def load_cols(path):
    if not path or not Path(path).exists():
        return None
    return set(pd.read_csv(path, nrows=0).columns)

def simulate_imputation(df):
    """Simulates three-pass imputation and returns fill counts per pass."""
    numeric_cols = [c for c in df.select_dtypes(include=['number']).columns
                    if 'exchange' not in c.lower()]
    if not numeric_cols:
        return 0, 0, 0, 0, 0, {}

    work = df[numeric_cols].copy()
    if 'HourUTC' in df.columns:
        work.index = pd.to_datetime(df['HourUTC'], utc=True)
        work = work.sort_index()

    original_missing = work.isna()
    total_missing    = int(original_missing.sum().sum())
    total_values     = int(work.size)

    try:
        p1 = work.interpolate(method='time', limit=INTERPOLATION_LIMIT,
                               limit_direction='both')
    except Exception:
        p1 = work.interpolate(method='linear', limit=INTERPOLATION_LIMIT,
                               limit_direction='both')

    still_p1  = p1.isna()
    filled_p1 = int((original_missing & ~still_p1).sum().sum())

    p2        = p1.ffill(limit=INTERPOLATION_LIMIT).bfill(limit=INTERPOLATION_LIMIT)
    still_p2  = p2.isna()
    filled_p2 = int((still_p1 & ~still_p2).sum().sum())

    mean_cols = {col: int(still_p2[col].sum())
                 for col in numeric_cols if still_p2[col].sum() > 0}
    filled_p3 = sum(mean_cols.values())

    return total_values, total_missing, filled_p1, filled_p2, filled_p3, mean_cols

def identify_dropped(df, rows):
    dropped_m, dropped_c = [], []
    for col in df.columns:
        if col in ALWAYS_KEEP or 'exchange' in col.lower():
            continue
        miss = df[col].isna().sum() / rows
        if miss > MISSING_THRESHOLD:
            dropped_m.append((col, miss))
            continue
        valid = df[col].dropna()
        if len(valid) > 0:
            top_r = valid.value_counts(normalize=True).iloc[0]
            if top_r > DOMINANCE_THRESHOLD:
                dropped_c.append((col, top_r))
    return dropped_m, dropped_c


# =====================================================================
# SOURCE AUDITS
# =====================================================================

def audit_price():
    header("PRICE DATA")

    sources = [
        ("Elspot Prices (pre-Oct 2025)",
         BASE_PATH / "Prices" / "Elspotprices_standardized.csv"),
        ("Day-Ahead Prices (post-Oct 2025)",
         BASE_PATH / "Prices" / "DayAheadPrices_standardized.csv"),
    ]

    grand_rows = grand_cols = grand_dropped = grand_missing = 0

    for name, path in sources:
        df = load_df(path, filter_dk=True)
        if df is None:
            print(f"\n  [{name}] — file not found, skipping.")
            continue

        rows = len(df)
        dm, dc = identify_dropped(df, rows)
        n_dropped = len(dm) + len(dc)
        df_clean = df.drop(columns=[c for c, _ in dm + dc] if dm + dc else [])
        tv, om, fp1, fp2, fp3, mc = simulate_imputation(df_clean)

        print(f"\n  {name}")
        divider("─", 70)
        row("Rows:", f"{rows:,}")
        row("Columns (raw):", len(df.columns))
        row("Columns dropped:", n_dropped)
        row("Columns retained:", len(df.columns) - n_dropped)
        row("Missing values:", f"{om:,}", pct(om, tv))
        row("Filled by interpolation:", f"{fp1:,}", pct(fp1, tv))
        row("Filled by ffill/bfill:", f"{fp2:,}", pct(fp2, tv))
        row("Filled by mean fallback:", f"{fp3:,}", pct(fp3, tv))

        if dm or dc:
            print(f"\n  Dropped columns:")
            for col, r in dm:
                print(f"    - {col}: {r:.1%} missing")
            for col, r in dc:
                print(f"    - {col}: constant in {r:.1%} of rows")
        if mc:
            print(f"  Mean fallback columns:")
            for col, cnt in sorted(mc.items(), key=lambda x: -x[1]):
                print(f"    - {col}: {cnt:,} values")

        grand_rows    += rows
        grand_cols    += len(df.columns)
        grand_dropped += n_dropped
        grand_missing += om

    print(f"\n  {'─' * 70}")
    print(f"  PRICE TOTALS")
    row("Combined rows:", f"{grand_rows:,}")
    row("Combined columns (raw):", grand_cols)
    row("Total dropped:", grand_dropped)
    row("Total missing values:", f"{grand_missing:,}")


def audit_prod_cons():
    header("PRODUCTION & CONSUMPTION DATA")

    path = BASE_PATH / "Prod_Cons" / "ProductionConsumptionSettlement_standardized.csv"
    df   = load_df(path, filter_dk=True)
    if df is None:
        print("  File not found, skipping.")
        return

    rows          = len(df)
    exchange_cols = [c for c in df.columns if 'exchange' in c.lower()]
    exch_missing  = int(df[exchange_cols].isna().sum().sum()) if exchange_cols else 0

    dm, dc    = identify_dropped(df, rows)
    n_dropped = len(dm) + len(dc)
    df_clean  = df.drop(columns=[c for c, _ in dm + dc] if dm + dc else [])
    tv, om, fp1, fp2, fp3, mc = simulate_imputation(df_clean)

    divider("─", 70)
    row("Rows:", f"{rows:,}")
    row("Columns (raw):", len(df.columns))
    row("  of which exchange columns:", len(exchange_cols))
    row("Columns dropped:", n_dropped)
    row("Columns retained:", len(df.columns) - n_dropped)
    print()
    row("Exchange columns — missing zeros filled:", f"{exch_missing:,}",
        pct(exch_missing, rows * len(exchange_cols)) if exchange_cols else "N/A")
    row("Other missing values:", f"{om:,}", pct(om, tv))
    row("  Filled by interpolation:", f"{fp1:,}", pct(fp1, tv))
    row("  Filled by ffill/bfill:", f"{fp2:,}", pct(fp2, tv))
    row("  Filled by mean fallback:", f"{fp3:,}", pct(fp3, tv))

    if dm or dc:
        print(f"\n  Dropped columns:")
        for col, r in dm:
            print(f"    - {col}: {r:.1%} missing")
        for col, r in dc:
            print(f"    - {col}: constant in {r:.1%} of rows")
    if mc:
        print(f"\n  Mean fallback columns:")
        for col, cnt in sorted(mc.items(), key=lambda x: -x[1]):
            print(f"    - {col}: {cnt:,} values")
    else:
        print(f"\n  Mean fallback: never triggered")


def audit_weather(region):
    header(f"WEATHER DATA — {region}")

    yearly_files = sorted(DMI_PROCESSED_DIR.glob(f"*_{region}_hourly.csv"))
    aligned_files = sorted(DMI_ALIGNED_DIR.glob(f"*_{region}_hourly_aligned.csv"))
    master_files  = sorted(ML_READY_DIR.glob(f"Master_Matrix_{region}_*.csv"))

    if not yearly_files:
        print(f"  No ProcessedZones files found for {region}.")
        return

    # Aggregate across all yearly files
    agg = dict(rows=0, raw_cols=0, dropped_m=0, dropped_c=0,
               tv=0, om=0, fp1=0, fp2=0, fp3=0)
    all_dropped_m, all_dropped_c = {}, {}
    all_mean_cols  = {}
    per_year_cols  = []

    for f in yearly_files:
        year = f.name.split('_')[0]
        df   = load_df(f)
        if df is None or df.empty:
            continue

        rows = len(df)
        dm, dc = identify_dropped(df, rows)
        df_clean = df.drop(columns=[c for c, _ in dm + dc] if dm + dc else [])
        tv, om, fp1, fp2, fp3, mc = simulate_imputation(df_clean)

        per_year_cols.append(set(df_clean.columns))
        agg["rows"]      += rows
        agg["raw_cols"]  += len(df.columns)
        agg["dropped_m"] += len(dm)
        agg["dropped_c"] += len(dc)
        agg["tv"]        += tv
        agg["om"]        += om
        agg["fp1"]       += fp1
        agg["fp2"]       += fp2
        agg["fp3"]       += fp3

        for col, r in dm:
            all_dropped_m.setdefault(col, []).append(year)
        for col, r in dc:
            all_dropped_c.setdefault(col, []).append(year)
        for col, cnt in mc.items():
            all_mean_cols[f"{year}/{col}"] = cnt

    # Cross-year intersection (stage 1→2)
    if per_year_cols:
        union        = per_year_cols[0].copy()
        intersection = per_year_cols[0].copy()
        for cols in per_year_cols[1:]:
            union        |= cols
            intersection &= cols
        dropped_by_intersection = union - intersection
    else:
        dropped_by_intersection = set()

    # Final aligned column count
    n_aligned = len(load_cols(aligned_files[0])) if aligned_files else "N/A"

    # Master matrix stats
    if master_files:
        mf          = master_files[0]
        master_cols = load_cols(mf)
        master_rows = sum(1 for _ in open(mf)) - 1
        lag_cols    = [c for c in (master_cols or []) if '_lag_' in c]
        target_cols = [c for c in (master_cols or []) if c.startswith('TARGET_')]
        time_cols   = [c for c in (master_cols or []) if any(
                        x in c for x in ['_sin', '_cos', 'hour', 'month',
                                         'dayofweek', 'dayofyear'])]
        n_master_cols = len(master_cols) if master_cols else "N/A"
    else:
        master_rows = n_master_cols = lag_cols = target_cols = time_cols = None

    divider("─", 70)
    row("Yearly files processed:", len(yearly_files))
    row("Total rows (raw):", f"{agg['rows']:,}")
    row("Columns per year (raw, avg):",
        f"{agg['raw_cols'] // len(yearly_files) if yearly_files else 0}")
    print()
    row("Dropped per-year (>90% missing):", agg["dropped_m"])
    row("Dropped per-year (>95% constant):", agg["dropped_c"])
    row("Dropped by cross-year intersection:", len(dropped_by_intersection))
    row("Columns in aligned output:", n_aligned)
    print()
    row("Total numeric cells:", f"{agg['tv']:,}")
    row("Originally missing:", f"{agg['om']:,}", pct(agg["om"], agg["tv"]))
    row("Filled by interpolation:", f"{agg['fp1']:,}", pct(agg["fp1"], agg["tv"]))
    row("Filled by ffill/bfill:", f"{agg['fp2']:,}", pct(agg["fp2"], agg["tv"]))
    row("Filled by mean fallback:", f"{agg['fp3']:,}", pct(agg["fp3"], agg["tv"]))

    if master_files:
        total_aligned_rows = sum(
            sum(1 for _ in open(f)) - 1 for f in aligned_files
        )
        rows_dropped = total_aligned_rows - master_rows
        print()
        row("Master matrix columns:", n_master_cols)
        row("  of which lag features:", len(lag_cols))
        row("  of which time encodings:", len(time_cols))
        row("  of which target columns:", len(target_cols))
        row("Master matrix rows:", f"{master_rows:,}")
        row("Rows dropped (NaN edges):", f"{rows_dropped:,}",
            pct(rows_dropped, total_aligned_rows))

    # Detail sections
    if all_dropped_m or all_dropped_c:
        print(f"\n  Dropped columns (per-year):")
        for col, years in sorted(all_dropped_m.items()):
            print(f"    - {col}: dropped in {', '.join(years)}")
        for col, years in sorted(all_dropped_c.items()):
            print(f"    - {col}: constant in {', '.join(years)}")

    if dropped_by_intersection:
        print(f"\n  Dropped by cross-year intersection:")
        for col in sorted(dropped_by_intersection):
            present = [i + 1 for i, cols in enumerate(per_year_cols) if col in cols]
            print(f"    - {col}: present in {len(present)}/{len(yearly_files)} years")

    if all_mean_cols:
        print(f"\n  Mean fallback triggered:")
        for key, cnt in sorted(all_mean_cols.items(), key=lambda x: -x[1]):
            print(f"    - {key}: {cnt:,} values")
    else:
        print(f"\n  Mean fallback: never triggered")


# =====================================================================
# MAIN
# =====================================================================

def main():
    divider()
    print("  PREPROCESSING OVERVIEW AUDIT")
    divider()

    audit_price()
    audit_prod_cons()
    audit_weather("DK1")
    audit_weather("DK2")

    print()
    divider()
    print("  AUDIT COMPLETE")
    divider()


if __name__ == "__main__":
    main()