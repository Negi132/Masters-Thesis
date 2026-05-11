import os
import pickle
from pathlib import Path

LOGS_DIR = Path("Experiment_Logs")
OUTPUT_FILE = "pkl_audit_report.txt"

def audit_pkl_files():
    print("Starting PKL File Audit...")
    print("-" * 60)

    if not LOGS_DIR.exists():
        print(f"[ERROR] Could not find '{LOGS_DIR}' directory.")
        print("Make sure you run this script from your ML_Pipeline directory.")
        return

    pkl_files = sorted(LOGS_DIR.glob("*.pkl"))

    if not pkl_files:
        print(f"[ERROR] No .pkl files found in {LOGS_DIR}")
        return

    print(f"Found {len(pkl_files)} pkl files.\n")

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("           PKL FILE AUDIT REPORT")
    report_lines.append("=" * 60)

    total_ok = 0
    total_empty = 0
    total_error = 0

    for f in pkl_files:
        report_lines.append(f"\n{f.name}")
        report_lines.append("-" * 50)

        try:
            with open(f, 'rb') as fh:
                data = pickle.load(fh)

            models = list(data.keys())

            for model_name in models:
                model_data = data[model_name]
                n_true = len(model_data.get('y_true', []))
                n_pred = len(model_data.get('y_pred', []))
                n_curves = len(model_data.get('learning_curves', []))

                if n_true > 0 and n_pred > 0:
                    status = "OK"
                    total_ok += 1
                else:
                    status = "EMPTY"
                    total_empty += 1

                line = f"  {model_name:<15} | Predictions: {n_true:>6} rows | Curves: {n_curves:>3} windows | [{status}]"
                report_lines.append(line)
                print(line)

        except Exception as e:
            msg = f"  [ERROR] Could not read file: {e}"
            report_lines.append(msg)
            print(msg)
            total_error += 1

    # Summary
    summary = [
        "\n" + "=" * 60,
        "SUMMARY",
        "=" * 60,
        f"  Total model entries with data:  {total_ok}",
        f"  Total empty model entries:      {total_empty}",
        f"  Total unreadable files:         {total_error}",
        "=" * 60
    ]

    for line in summary:
        report_lines.append(line)
        print(line)

    # Save report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"\nFull report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    audit_pkl_files()