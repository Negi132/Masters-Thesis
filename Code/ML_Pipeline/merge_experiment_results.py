"""
MERGE EXPERIMENT RESULTS
=========================
Merges new experiment results from another machine into the local clean CSV.

Inputs:
  - LOCAL_CSV:  the existing experiment_results_clean.csv on this machine
  - INCOMING_CSV: the CSV transferred from the other machine

Logic:
  1. Load both CSVs, tolerating column-order differences.
  2. Concatenate them.
  3. Sort by Timestamp ascending so the newest rows are last.
  4. Drop duplicates on (Experiment, Model, Region, Target) keeping the last
     (= newest timestamp). This matches cleanDuplicateCSV.py exactly.
  5. Sort the final output by (Experiment, Model) for stable diffs.
  6. Write to OUTPUT_CSV. Backup the original LOCAL_CSV first.

Run from the directory where the CSVs live (or adjust paths below).
"""

import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

# =====================================================================
# CONFIGURATION
# =====================================================================
LOCAL_CSV    = Path("experiment_results_clean.csv")
INCOMING_CSV = Path("experiment_results_incoming.csv")
OUTPUT_CSV   = Path("experiment_results_clean.csv")  # overwrite local
BACKUP_DIR   = Path("csv_backups")

DEDUP_KEYS = ['Experiment', 'Model', 'Region', 'Target']


def main():
    print("=" * 70)
    print("  EXPERIMENT RESULTS MERGE")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Validate inputs
    # ---------------------------------------------------------------
    if not LOCAL_CSV.exists():
        print(f"[ERROR] Local CSV not found: {LOCAL_CSV}")
        return
    if not INCOMING_CSV.exists():
        print(f"[ERROR] Incoming CSV not found: {INCOMING_CSV}")
        return

    # ---------------------------------------------------------------
    # 2. Backup the local CSV before doing anything destructive
    # ---------------------------------------------------------------
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"experiment_results_clean_backup_{timestamp}.csv"
    shutil.copy2(LOCAL_CSV, backup_path)
    print(f"\n  Backed up local CSV to: {backup_path}")

    # ---------------------------------------------------------------
    # 3. Load both CSVs
    # ---------------------------------------------------------------
    df_local    = pd.read_csv(LOCAL_CSV, sep=None, engine='python')
    df_incoming = pd.read_csv(INCOMING_CSV, sep=None, engine='python')

    print(f"\n  Local rows:    {len(df_local):>6}")
    print(f"  Incoming rows: {len(df_incoming):>6}")

    # ---------------------------------------------------------------
    # 4. Tolerate column-order differences. Warn if columns mismatch.
    # ---------------------------------------------------------------
    local_cols    = set(df_local.columns)
    incoming_cols = set(df_incoming.columns)

    only_in_local    = local_cols - incoming_cols
    only_in_incoming = incoming_cols - local_cols

    if only_in_local:
        print(f"\n  [WARN] Columns only in local CSV: {sorted(only_in_local)}")
        print(f"         These will be NaN for incoming rows.")
    if only_in_incoming:
        print(f"\n  [WARN] Columns only in incoming CSV: {sorted(only_in_incoming)}")
        print(f"         These will be NaN for local rows.")

    # Use the local column order as the canonical order, append any new columns
    # from incoming at the end so nothing silently disappears.
    canonical_cols = list(df_local.columns) + sorted(only_in_incoming)

    # ---------------------------------------------------------------
    # 5. Verify the dedup keys exist in both CSVs
    # ---------------------------------------------------------------
    missing_keys_local    = [k for k in DEDUP_KEYS if k not in df_local.columns]
    missing_keys_incoming = [k for k in DEDUP_KEYS if k not in df_incoming.columns]
    if missing_keys_local or missing_keys_incoming:
        print(f"\n[ERROR] Dedup keys missing:")
        if missing_keys_local:
            print(f"  Local CSV missing:    {missing_keys_local}")
        if missing_keys_incoming:
            print(f"  Incoming CSV missing: {missing_keys_incoming}")
        return

    # ---------------------------------------------------------------
    # 6. Concatenate
    # ---------------------------------------------------------------
    df_combined = pd.concat([df_local, df_incoming], ignore_index=True, sort=False)
    # Reindex to canonical column order
    df_combined = df_combined.reindex(columns=canonical_cols)
    print(f"\n  Combined rows (pre-dedup): {len(df_combined):>6}")

    # ---------------------------------------------------------------
    # 7. Sort by timestamp ASCENDING so the newest row appears last
    #    for each (Experiment, Model, Region, Target) group.
    #    Then drop_duplicates with keep='last' = newest timestamp wins.
    # ---------------------------------------------------------------
    if 'Timestamp' not in df_combined.columns:
        print("\n[ERROR] No 'Timestamp' column found — cannot determine which row is newest.")
        return

    df_combined['Timestamp'] = pd.to_datetime(
        df_combined['Timestamp'], errors='coerce'
    )

    # Rows with unparseable timestamps go to the BOTTOM of their group
    # (treated as oldest) so a parseable timestamp wins over an unparseable one.
    df_combined = df_combined.sort_values(
        'Timestamp', ascending=True, na_position='first'
    )

    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=DEDUP_KEYS, keep='last')
    after = len(df_combined)
    print(f"  Dropped duplicates:        {before - after:>6}")
    print(f"  Final rows:                {after:>6}")

    # ---------------------------------------------------------------
    # 8. Diagnostic: how many rows did the incoming CSV actually contribute?
    # ---------------------------------------------------------------
    # We re-derive this from a marker — easier just to compare key sets.
    local_keys    = set(map(tuple, df_local[DEDUP_KEYS].values))
    incoming_keys = set(map(tuple, df_incoming[DEDUP_KEYS].values))
    new_keys      = incoming_keys - local_keys
    overlap_keys  = incoming_keys & local_keys

    print(f"\n  Incoming contribution breakdown:")
    print(f"    Brand-new experiments:        {len(new_keys):>6}")
    print(f"    Overlapping (newer wins):     {len(overlap_keys):>6}")

    # ---------------------------------------------------------------
    # 9. Final sort + write
    # ---------------------------------------------------------------
    df_combined = df_combined.sort_values(['Experiment', 'Model'])
    df_combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Merged CSV written to: {OUTPUT_CSV}")
    print("=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"\n  Next steps:")
    print(f"    1. Inspect the merged CSV manually if you want.")
    print(f"    2. Run the master_plotter to regenerate plots.")
    print(f"    3. If anything looks wrong, restore from: {backup_path}")


if __name__ == "__main__":
    main()
