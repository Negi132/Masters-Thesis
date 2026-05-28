"""
Data Snapshot Table Generator
==============================
Generates LaTeX tables showing raw vs preprocessed snapshots of each
data source plus the integrated master matrix. Replaces the placeholder
Table I in the paper with real values from your actual data files.

Output: four .tex files in Tables/
  - data_snapshot_weather.tex      (Raw DMI weather row vs. master row)
  - data_snapshot_pc.tex            (Raw P&C row vs. master row)
  - data_snapshot_price.tex         (Raw price row vs. master row)
  - data_snapshot_master.tex        (Master matrix row only, all groups)

Configuration block at the top lets you adjust:
  - Paths to raw data files
  - Which timestamp to snapshot (defaults to a sensible mid-data point)
  - How many columns to show per source
  - Whether the output should be wrapped in \\begin{landscape}

The script uses real timestamps and real values. If a column does not
exist in your data (e.g. you renamed it during preprocessing), the row
shows "--" rather than fabricating a value.
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path

# Allow imports from the ML_Pipeline package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# CONFIGURATION
# =====================================================================
# Adjust these paths to match your actual file layout. Paths are
# relative to the directory this script lives in.

# Raw weather: a directory containing per-year hourly CSV files
# (e.g. 2015_DK1_hourly.csv, 2016_DK1_hourly.csv, ...).
# The script will concatenate every file matching WEATHER_PATTERN.
RAW_WEATHER_DIR     = "../Data_Engineering/Data/DMI/ProcessedZones"
RAW_WEATHER_PATTERN = "*_DK1_hourly.csv"

# Raw production & consumption: single CSV containing both DK1 and DK2.
# The script filters to DK1 only.
RAW_PC_PATH = "../Data_Engineering/Data/Prod_Cons/ProductionConsumptionSettlement.csv"

# Raw price: two files concatenated as in your IV-A-3 preprocessing.
# Elspotprices.csv (pre-Oct 2025) + DayAheadPrices.csv (post-Oct 2025).
# The script handles both and concatenates them in chronological order.
RAW_PRICE_PATHS = [
    "../Data_Engineering/Data/Prices/Elspotprices.csv",
    "../Data_Engineering/Data/Prices/DayAheadPrices.csv",
]

# Master matrix to compare against. 24h horizon is preferred when
# available (contains TARGET_Delta_24h which gives a more complete
# cross-section), but 0h is acceptable.
MASTER_MATRIX_PATH = "../Data_Engineering/Data/ML_Ready_Data/Master_Matrix_DK1_Horizon24h.csv"
# Fallback master matrix if the preferred one is missing
MASTER_MATRIX_FALLBACK_PATH = "../Data_Engineering/Data/ML_Ready_Data/Master_Matrix_DK1_Horizon0h.csv"

# Filter raw P&C and price data to this region. Use the region label
# found in the raw EnergiNet CSV's PriceArea (or equivalent) column.
RAW_REGION = "DK1"

# Where to drop the generated .tex files (sibling of this script)
OUTPUT_DIR = "Tables"

# Wrap each generated table in \begin{landscape}/\end{landscape}?
USE_LANDSCAPE = True

# Pin a specific UTC hour to snapshot. If None, the script auto-picks
# a timestamp that exists in all sources AND the master matrix.
# Format: ISO 8601 string, e.g. "2024-06-15T12:00:00+00:00"
SNAPSHOT_TIMESTAMP = None

# Maximum columns per side of the comparison when wrapping is OFF.
# Tables wider than this get truncated to the first N most informative
# columns. Ignored when WRAP_WIDE_TABLES is True (then COLS_PER_CHUNK
# controls the layout).
MAX_COLS_PER_TABLE = 10

# When True, tables that have more columns than COLS_PER_CHUNK are
# NOT truncated. Instead, every column is shown, but split into
# multiple stacked sub-tables inside the same table* float. A
# continuation indicator (down-arrow) separates the chunks. Each
# chunk repeats the timestamp as its first column so the reader can
# orient themselves.
WRAP_WIDE_TABLES = True

# Number of data columns per chunk (excluding the timestamp). Choose
# small enough that one chunk fits on a landscape page. 8 is a safe
# default for most paper layouts. Per-source overrides may be set in
# COLS_PER_CHUNK_OVERRIDES below when a specific table needs to be
# narrower (e.g. when its column names are long enough that 8 columns
# overflow even in landscape).
COLS_PER_CHUNK = 8

# Optional per-source overrides for COLS_PER_CHUNK. Keys are source
# labels matching the lowercase identifier used in the output file
# names (without the "data_snapshot_" prefix). Use this when one
# table's columns are wider than the others.
COLS_PER_CHUNK_OVERRIDES = {
    "pc": 7,        # production & consumption has long MWh-style names
}

# Per-source override of COLS_PER_CHUNK. Sources with longer column
# names need fewer columns per chunk to fit on a landscape page. Set
# to None to use the global COLS_PER_CHUNK default.
COLS_PER_CHUNK_PER_SOURCE = {
    "weather": None,   # uses COLS_PER_CHUNK
    "pc":      7,      # production/consumption columns are wider, use one fewer
    "price":   None,
    "master":  None,
}

# When True, the master matrix table excludes any column that already
# appears in a per-source table (weather, P&C, price). What remains
# is the "integration glue" - Time features, TARGET columns, imputation
# flags, etc. Drastically reduces the master matrix table size and
# avoids duplicating data already shown in the per-source tables.
MASTER_EXCLUDE_GROUPS_SHOWN_ELSEWHERE = True

# What to render between chunks. "↓" is the cleanest if your document
# supports utf-8 (modern LaTeX defaults to utf-8 input). If you get
# a compile error about an unknown character, switch to
# CONTINUATION_MARKER = "(continued below)"
CONTINUATION_MARKER = "$\\downarrow$ (continued)"

# Vertical space between the Raw and Preprocessed sections of a
# comparison table. "1em" is roughly one line of text; "2.5em" is
# roughly two and a half lines. Bump up if the two sections feel
# too tight; bump down if they feel too separated.
SECTION_SEPARATOR_VSPACE = "2.5em"

# When True, long column-name headers and long timestamps are split
# across two lines inside each cell using \makecell. This makes wide
# tables more compact horizontally at the cost of slight vertical
# growth. Requires \usepackage{makecell} in the LaTeX preamble.
WRAP_LONG_HEADERS = True

# Headers shorter than this character count are NOT wrapped (no point
# in two-lining a header like "Time" or "MAE"). Counted against the
# original column name BEFORE LaTeX-escaping.
LONG_HEADER_THRESHOLD = 14


# =====================================================================
# LATEX ESCAPING (same as generate_summary_tables.py)
# =====================================================================
_LATEX_SPECIALS = {
    '\\': r'\textbackslash{}',
    '{':  r'\{', '}':  r'\}',
    '_':  r'\_', '%':  r'\%',
    '$':  r'\$', '&':  r'\&',
    '#':  r'\#', '^':  r'\^{}',
    '~':  r'\~{}',
}

def latex_escape(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "--"
    s = str(s)
    out = s.replace('\\', _LATEX_SPECIALS['\\'])
    for ch, repl in _LATEX_SPECIALS.items():
        if ch == '\\':
            continue
        out = out.replace(ch, repl)
    return out


def format_value(v, decimals=2):
    """Render a value for a table cell with sensible formatting."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    if isinstance(v, float):
        # Use fewer decimals for very large values, more for tiny ones
        if abs(v) >= 1000:
            return f"{v:.0f}"
        elif abs(v) >= 1:
            return f"{v:.{decimals}f}"
        else:
            return f"{v:.4f}"
    return latex_escape(v)


def _wrap_long_name(name):
    """Split a long column name across two lines at the most central
    underscore, returning a \\makecell-formatted string. Returns the
    unwrapped (escaped) name if WRAP_LONG_HEADERS is off or if the
    name is short enough not to need wrapping."""
    if not WRAP_LONG_HEADERS:
        return latex_escape(name)
    if len(str(name)) <= LONG_HEADER_THRESHOLD:
        return latex_escape(name)

    # Find the underscore closest to the middle of the original name
    name_str = str(name)
    underscores = [i for i, ch in enumerate(name_str) if ch == '_']
    if not underscores:
        # No natural break point - leave it alone rather than splitting mid-word
        return latex_escape(name)

    mid = len(name_str) / 2
    best = min(underscores, key=lambda i: abs(i - mid))
    left = name_str[:best]
    right = name_str[best + 1:]  # drop the underscore we broke at
    return ("\\makecell[l]{" + latex_escape(left) +
            " \\\\ \\_" + latex_escape(right) + "}")


def _wrap_long_timestamp(ts):
    """Split a timestamp at the space between date and time so it
    occupies two lines via \\makecell. Returns the unwrapped (escaped)
    string if WRAP_LONG_HEADERS is off."""
    if not WRAP_LONG_HEADERS:
        return latex_escape(str(ts))
    s = str(ts)
    if ' ' in s:
        date_part, _, time_part = s.partition(' ')
        return ("\\makecell[l]{" + latex_escape(date_part) +
                " \\\\ " + latex_escape(time_part) + "}")
    return latex_escape(s)


# =====================================================================
# CORE LOADERS
# =====================================================================
def find_timestamp_column(df):
    """Look for a timestamp column under any of the common names used
    in your raw files. Returns the column name or None."""
    # Order matters: prefer UTC over local time
    for candidate in ['HourUTC', 'HourDK', 'Hour', 'Timestamp',
                      'datetime', 'DateTime', 'time']:
        if candidate in df.columns:
            return candidate
    return None


def find_region_column(df):
    """Look for a region/zone column under any of the common names."""
    for candidate in ['PriceArea', 'Region', 'Zone', 'BiddingZone']:
        if candidate in df.columns:
            return candidate
    return None


def normalise_timestamp(df, label):
    """Renames the discovered timestamp column to HourUTC and parses it
    to datetime[utc]. Returns the modified df or None if no column found."""
    ts_col = find_timestamp_column(df)
    if ts_col is None:
        print(f"  [WARN] {label}: no recognisable timestamp column. "
              f"Available columns: {list(df.columns)[:8]}...")
        return None
    if ts_col != 'HourUTC':
        df = df.rename(columns={ts_col: 'HourUTC'})
    df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True, errors='coerce')
    df = df.dropna(subset=['HourUTC']).reset_index(drop=True)
    return df


def filter_to_region(df, label):
    """Filter the DataFrame to RAW_REGION rows. Returns df unchanged if
    no region column is found (some sources have no region column)."""
    region_col = find_region_column(df)
    if region_col is None:
        return df
    before = len(df)
    df = df[df[region_col] == RAW_REGION].copy()
    print(f"  {label}: filtered {before:,} -> {len(df):,} rows for region={RAW_REGION}")
    return df


def safe_read_csv(path, label):
    """Read one CSV with timestamp normalisation. Returns df or None."""
    if not os.path.exists(path):
        print(f"  [WARN] {label} not found at {path}")
        return None
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception as e:
        print(f"  [WARN] Could not read {label} ({path}): {e}")
        return None
    return normalise_timestamp(df, label)


def load_raw_weather():
    """Concatenate per-year weather files matching WEATHER_PATTERN
    under RAW_WEATHER_DIR. Returns one DataFrame sorted by HourUTC."""
    if not os.path.isdir(RAW_WEATHER_DIR):
        print(f"  [WARN] Weather directory not found: {RAW_WEATHER_DIR}")
        return None
    pattern = os.path.join(RAW_WEATHER_DIR, RAW_WEATHER_PATTERN)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  [WARN] No weather files match {pattern}")
        return None
    print(f"  Loading {len(files)} weather files from {RAW_WEATHER_DIR}/")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=None, engine='python')
        except Exception as e:
            print(f"    Skipping {os.path.basename(f)}: {e}")
            continue
        df = normalise_timestamp(df, os.path.basename(f))
        if df is not None:
            frames.append(df)
    if not frames:
        print(f"  [WARN] No usable weather files.")
        return None
    df_all = pd.concat(frames, axis=0, ignore_index=True)
    df_all = df_all.sort_values('HourUTC').drop_duplicates(
        subset='HourUTC', keep='last').reset_index(drop=True)
    print(f"  Combined weather: {len(df_all):,} rows, "
          f"{df_all.shape[1]} columns")
    return df_all


def load_raw_pc():
    """Load the production & consumption CSV and filter to RAW_REGION."""
    df = safe_read_csv(RAW_PC_PATH, "Raw P&C")
    if df is None:
        return None
    df = filter_to_region(df, "Raw P&C")
    return df


def load_raw_price():
    """Concatenate Elspotprices.csv and DayAheadPrices.csv, filter to
    RAW_REGION, and discard non-hourly rows (matches your IV-A-3 logic)."""
    frames = []
    for path in RAW_PRICE_PATHS:
        df = safe_read_csv(path, f"Price ({os.path.basename(path)})")
        if df is None:
            continue
        df = filter_to_region(df, f"Price ({os.path.basename(path)})")
        # Keep only on-the-hour rows (drop quarter-hour data introduced
        # in October 2025 by Nord Pool's format change)
        before = len(df)
        df = df[df['HourUTC'].dt.minute == 0].reset_index(drop=True)
        if len(df) != before:
            print(f"    Dropped {before - len(df):,} non-hourly rows")
        frames.append(df)
    if not frames:
        return None
    df_all = pd.concat(frames, axis=0, ignore_index=True)
    df_all = df_all.sort_values('HourUTC').drop_duplicates(
        subset='HourUTC', keep='last').reset_index(drop=True)
    print(f"  Combined price: {len(df_all):,} rows")
    return df_all


def load_master_matrix():
    """Try the preferred horizon, fall back to the alternative if missing."""
    for path in [MASTER_MATRIX_PATH, MASTER_MATRIX_FALLBACK_PATH]:
        if os.path.exists(path):
            print(f"  Loading master matrix: {path}")
            df = pd.read_csv(path, sep=None, engine='python')
            return normalise_timestamp(df, "Master matrix")
    print(f"  [FATAL] No master matrix found at either:")
    print(f"            {MASTER_MATRIX_PATH}")
    print(f"            {MASTER_MATRIX_FALLBACK_PATH}")
    return None


def pick_snapshot_timestamp(dataframes):
    """Picks a timestamp present in every supplied DataFrame.
    Uses the median of the intersection of timestamps so the snapshot
    is reasonably representative."""
    if SNAPSHOT_TIMESTAMP is not None:
        return pd.to_datetime(SNAPSHOT_TIMESTAMP, utc=True)

    ts_sets = []
    for df in dataframes:
        if df is None or 'HourUTC' not in df.columns:
            continue
        ts_sets.append(set(df['HourUTC'].dropna()))

    if not ts_sets:
        print("  [WARN] No HourUTC columns found - cannot pick a snapshot.")
        return None

    common = ts_sets[0]
    for s in ts_sets[1:]:
        common &= s

    if not common:
        print("  [WARN] No timestamps shared across all sources.")
        return None

    sorted_common = sorted(common)
    pick = sorted_common[len(sorted_common) // 2]
    print(f"  Snapshot timestamp: {pick}")
    return pick


def row_at(df, ts):
    """Returns a single row at the given timestamp as a Series, or None."""
    if df is None or 'HourUTC' not in df.columns:
        return None
    match = df[df['HourUTC'] == ts]
    if match.empty:
        return None
    return match.iloc[0]


# =====================================================================
# COLUMN SELECTION
# =====================================================================
def pick_columns_for_source(raw_df, master_df, source_groups):
    """
    Returns a list of columns to display for this source.

    The selection prefers columns that exist in BOTH the raw and master
    matrices when both are available. Falls back to all raw columns if
    no master is available.

    source_groups: list of group names from config.COL_GROUPS that
    belong to this source (e.g. ['Weather', 'WeatherLags'] for weather).
    """
    try:
        from ML_Pipeline import config
        master_cols_for_source = set()
        for g in source_groups:
            master_cols_for_source.update(config.COL_GROUPS.get(g, []))
    except Exception:
        master_cols_for_source = set()

    candidates = []

    # Prefer columns present in both raw and master
    if raw_df is not None and master_df is not None:
        for c in raw_df.columns:
            if c == 'HourUTC':
                continue
            if c in master_df.columns:
                candidates.append(c)

    # If we found very few, fall back to raw-only columns
    if len(candidates) < 3 and raw_df is not None:
        candidates = [c for c in raw_df.columns if c != 'HourUTC']

    # Truncate only when wrapping is OFF. With wrapping ON, all columns
    # are kept and split into chunks at render time.
    if WRAP_WIDE_TABLES:
        return candidates
    return candidates[:MAX_COLS_PER_TABLE - 1]


def pick_master_matrix_columns(master_df):
    """For the master matrix table, pick a representative cross-section.

    With MASTER_EXCLUDE_GROUPS_SHOWN_ELSEWHERE=True (the default), columns
    belonging to groups already displayed in per-source tables (weather,
    P&C, price) are excluded. What remains is the integration glue -
    Time features, TARGET columns, imputation flags, and anything else
    that only exists post-aggregation.

    With the exclusion off, every column group is included, the way the
    earlier version of this script did. Useful if you want a single
    omnibus table with everything in it.
    """
    # Groups whose contents are already displayed in per-source tables
    GROUPS_SHOWN_ELSEWHERE = {
        "Weather", "WeatherLags",
        "Grid", "GridLags", "GridExchange", "GridExchangeLags",
        "Prices", "PriceLags",
    }

    try:
        from ML_Pipeline import config
    except Exception:
        print("  [WARN] config not importable - using first N master columns.")
        cols_to_keep = [c for c in master_df.columns if c != 'HourUTC']
        if not WRAP_WIDE_TABLES:
            cols_to_keep = cols_to_keep[:MAX_COLS_PER_TABLE - 1]
        return cols_to_keep

    picks = []
    seen = set()
    # Order matters - keep this consistent with how groups are introduced
    # in your paper's Section V-A.
    group_order = ["Time", "Weather", "WeatherLags", "Grid", "GridLags",
                   "GridExchange", "GridExchangeLags", "Prices", "PriceLags"]

    for group_name in group_order:
        # Skip groups already shown in per-source tables (when enabled)
        if MASTER_EXCLUDE_GROUPS_SHOWN_ELSEWHERE and \
                group_name in GROUPS_SHOWN_ELSEWHERE:
            continue

        cols = config.COL_GROUPS.get(group_name, [])
        # When wrapping is on take every column in the group; otherwise just one.
        keep = cols if WRAP_WIDE_TABLES else cols[:1]
        for c in keep:
            if c in master_df.columns and c not in seen:
                picks.append(c)
                seen.add(c)
        if not WRAP_WIDE_TABLES and len(picks) >= MAX_COLS_PER_TABLE - 1:
            return picks

    # Add TARGET columns - these only exist in the master matrix and
    # are conceptually "integration glue" alongside the time features.
    for c in master_df.columns:
        if c == 'HourUTC':
            continue
        if c.startswith('TARGET_') and c not in seen:
            picks.append(c)
            seen.add(c)

    # Add imputation flags - same rationale.
    for c in master_df.columns:
        if c == 'HourUTC':
            continue
        if c.endswith('_imputed') and c not in seen:
            picks.append(c)
            seen.add(c)
            if not WRAP_WIDE_TABLES and len(picks) >= MAX_COLS_PER_TABLE - 1:
                break

    return picks


# =====================================================================
# TABLE WRITER
# =====================================================================
def _chunk_columns(columns, chunk_size):
    """Split a column list into chunks of at most chunk_size columns each.
    Returns a list of column lists."""
    if chunk_size <= 0:
        return [columns]
    return [columns[i:i + chunk_size]
            for i in range(0, len(columns), chunk_size)]


def _emit_section_block(lines, section_title, columns, row, snapshot_ts,
                         is_first_section, chunk_size=None):
    """Emit one section (e.g. 'Raw Weather Data' or 'Preprocessed Weather Data')
    inside the table* float, splitting wide column lists into stacked
    tabular blocks connected by continuation markers.

    `lines` is appended to in place.
    `is_first_section` controls whether to add a section-separator before
    this block (used to put extra vertical space between the raw and
    preprocessed sections of a comparison table).
    `chunk_size` overrides COLS_PER_CHUNK for this call - used when a
    specific source needs a narrower or wider chunk than the global default.
    """
    # Section separator from the previous section
    if not is_first_section:
        lines.append(f"\\vspace{{{SECTION_SEPARATOR_VSPACE}}}")

    # Section title
    lines.append(f"\\noindent\\textbf{{{section_title}}}\\par")
    lines.append("\\vspace{0.3em}")

    # Decide whether to chunk
    effective_chunk_size = chunk_size if chunk_size is not None else COLS_PER_CHUNK
    if WRAP_WIDE_TABLES and len(columns) > effective_chunk_size:
        chunks = _chunk_columns(columns, effective_chunk_size)
    else:
        chunks = [columns]

    ts_cell = _wrap_long_timestamp(snapshot_ts)

    for chunk_idx, chunk_cols in enumerate(chunks):
        ncols_chunk = 1 + len(chunk_cols)
        col_spec = "@{} l " + " ".join(["l"] * len(chunk_cols)) + " @{}"

        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row - long names get split across two lines via \makecell
        header_cells = ["\\textbf{Timestamp}"] + \
                       [f"\\textbf{{{_wrap_long_name(c)}}}" for c in chunk_cols]
        lines.append(" & ".join(header_cells) + " \\\\")
        lines.append("\\midrule")

        # Data row (or "--" stub when no source row is available)
        if row is None:
            value_cells = ["--"] * ncols_chunk
        else:
            value_cells = [ts_cell]
            for c in chunk_cols:
                v = row.get(c) if hasattr(row, 'get') else None
                value_cells.append(format_value(v))
        lines.append(" & ".join(value_cells) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        # Force a paragraph break so any following content (continuation
        # marker, next section title, or end of table) starts on a fresh
        # line instead of flowing inline with the last tabular cell.
        lines.append("\\par")

        # Continuation marker between consecutive chunks of the same section
        if chunk_idx < len(chunks) - 1:
            lines.append("")
            lines.append(f"\\vspace{{0.2em}}")
            lines.append(f"\\centerline{{{CONTINUATION_MARKER}}}")
            lines.append(f"\\par\\vspace{{0.2em}}")
            lines.append("")


def write_snapshot_table(source_name, raw_row, master_row,
                         columns, snapshot_ts, label_suffix,
                         out_path, source_key=None):
    """Writes a side-by-side raw vs preprocessed snapshot for one source.

    When the column list is wider than COLS_PER_CHUNK and WRAP_WIDE_TABLES
    is on, each section is rendered as multiple stacked tabular blocks
    connected by a continuation marker, instead of one wide table.

    If `raw_row` is None, only the preprocessed (master matrix) snapshot
    is written.

    `source_key` (optional) is the lowercase identifier matching keys in
    COLS_PER_CHUNK_OVERRIDES. When supplied, an override there takes
    precedence over the global COLS_PER_CHUNK for this table only.
    """
    # Resolve chunk size override for this source, if any
    chunk_size = COLS_PER_CHUNK_OVERRIDES.get(source_key) if source_key else None

    lines = []
    lines.append(f"% Auto-generated by generate_data_snapshot_tables.py")
    lines.append(f"% Snapshot timestamp: {snapshot_ts}")
    lines.append("% Requires \\usepackage{booktabs}")
    if WRAP_LONG_HEADERS:
        lines.append("% Requires \\usepackage{makecell} for two-line column headers")
    if USE_LANDSCAPE:
        lines.append("% Requires \\usepackage{pdflscape} or \\usepackage{lscape}")
        lines.append("")
        lines.append("\\begin{landscape}")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Snapshot of a data record from {source_name}, "
                 f"at timestamp {snapshot_ts}. {label_suffix}}}")
    lines.append(f"\\label{{tab:snapshot-{source_name.lower().replace(' ', '-')}}}")
    lines.append("")

    # Section 1: Raw (if provided)
    is_first = True
    if raw_row is not None:
        _emit_section_block(lines, f"Raw {source_name}", columns,
                            raw_row, snapshot_ts,
                            is_first_section=is_first,
                            chunk_size=chunk_size)
        is_first = False

    # Section 2: Preprocessed (master matrix slice). Used standalone for
    # the master-only table; used as the second half for comparison tables.
    section_title = ("Preprocessed " + source_name) if raw_row is not None \
                    else f"Master Matrix Snapshot ({source_name})"
    _emit_section_block(lines, section_title, columns,
                        master_row, snapshot_ts,
                        is_first_section=is_first,
                        chunk_size=chunk_size)

    lines.append("\\end{table*}")
    if USE_LANDSCAPE:
        lines.append("\\end{landscape}")

    with open(out_path, 'w') as f:
        f.write("\n".join(lines))


# =====================================================================
# MAIN
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  DATA SNAPSHOT TABLE GENERATOR")
    print("=" * 60)

    # 1. Load all the data
    df_weather = load_raw_weather()
    df_pc      = load_raw_pc()
    df_price   = load_raw_price()
    df_master  = load_master_matrix()

    if df_master is None:
        print("\n[FATAL] Master matrix is required. Cannot proceed.")
        return

    # 2. Pick a snapshot timestamp that exists everywhere
    snapshot_ts = pick_snapshot_timestamp(
        [df_weather, df_pc, df_price, df_master]
    )
    if snapshot_ts is None:
        print("\n[FATAL] Could not pick a snapshot timestamp.")
        return

    # 3. Extract rows at that timestamp from each source
    weather_row = row_at(df_weather, snapshot_ts)
    pc_row      = row_at(df_pc, snapshot_ts)
    price_row   = row_at(df_price, snapshot_ts)
    master_row  = row_at(df_master, snapshot_ts)

    if master_row is None:
        print("\n[FATAL] Master matrix has no row at the chosen timestamp.")
        return

    # 4. Generate one table per source
    sources = [
        ("Weather Data",        weather_row, ["Weather", "WeatherLags"],
         "Raw values are station-aggregated hourly DMI measurements for "
         "the DK1 region. Preprocessed columns are hourly mean values. "
         "Weather\_Lag features are not depicted, but are identical values from 24 hours before.",
         "weather"),
        ("Production and Consumption Data", pc_row,
         ["Grid", "GridLags", "GridExchange", "GridExchangeLags"],
         "Raw values are hourly settlement-period production and consumption "
         "figures from EnergiNet. Preprocessed columns retain the original "
         "values with missing energy exchanges encoded as zero.",
         "pc"),
        ("Price Data",          price_row,   ["Prices", "PriceLags"],
         "Raw values are hourly day-ahead spot prices from Nord Pool via "
         "EnergiNet. Preprocessing adds 24-hour, 48-hour, and "
         "168-hour lagged price columns plus the price-delta target variants, though they are not depicted here.",
         "price"),
    ]

    for source_name, raw_row, groups, caption_suffix, file_label in sources:
        if raw_row is None:
            print(f"\n[SKIP] {source_name}: no raw row at this timestamp.")
            continue
        cols = pick_columns_for_source(
            df_weather if file_label == "weather" else
            df_pc      if file_label == "pc"      else
            df_price,
            df_master, groups
        )
        if not cols:
            print(f"\n[SKIP] {source_name}: no columns to show.")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"data_snapshot_{file_label}.tex")
        write_snapshot_table(
            source_name, raw_row, master_row, cols,
            snapshot_ts, caption_suffix, out_path,
            source_key=file_label,
        )
        print(f"  Wrote {os.path.basename(out_path)} "
              f"({len(cols)} columns)")

    # 5. Master matrix table - shows ONLY columns not displayed in
    # the per-source tables. This includes Time features, TARGET
    # columns, and imputation flags - the "integration glue" that
    # only exists after raw sources are joined into the master matrix.
    print()
    master_cols = pick_master_matrix_columns(df_master)
    if master_cols:
        out_path = os.path.join(OUTPUT_DIR, "data_snapshot_master.tex")
        if MASTER_EXCLUDE_GROUPS_SHOWN_ELSEWHERE:
            caption = (
                "Master matrix columns that are introduced during "
                "integration and therefore do not appear in the "
                "preceding per-source tables. These include cyclical "
                "time encodings, prediction target columns, and "
                "imputation flags that mark synthetic values produced "
                "during preprocessing."
            )
        else:
            caption = (
                "A representative cross-section of the integrated "
                "master matrix showing every feature group at the "
                "snapshot timestamp."
            )
        write_snapshot_table(
            "Master Matrix",
            None,  # no raw side for the master table
            master_row, master_cols, snapshot_ts,
            caption, out_path,
            source_key="master",
        )
        print(f"  Wrote {os.path.basename(out_path)} "
              f"({len(master_cols)} columns)")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
