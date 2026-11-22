"""
Automated workflow:  drift check  ->  retrain  ->  verification gate.

1. Compares a batch of production customers with the training data (PSI).
2. If at least one feature is in ALERT, it trains a new candidate and runs the gate.
   The gate promotes it ONLY if AUC and F1 both beat the champion.
3. If nothing drifted, it stops: the champion keeps serving.

Usage (run from the project root, MLflow server running):
    python training/workflow.py                                  # uses data/production_drifted.csv
    python training/workflow.py --batch data/production_stable.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                 # so 'monitoring' can be imported
sys.path.insert(0, str(ROOT / "training"))    # so 'retrain_gate' can be imported

from monitoring.psi import psi_report         # noqa: E402
from retrain_gate import run_gate             # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="data/production_drifted.csv")
    args = parser.parse_args()

    reference = pd.read_csv(ROOT / "data" / "reference.csv")
    batch = pd.read_csv(ROOT / args.batch)[list(reference.columns)]

    print(f"STEP 1 - Drift check: {len(batch)} production customers vs {len(reference)} training customers")
    report = psi_report(reference, batch)
    for _, row in report.iterrows():
        print(f"  [{row['status']:<7}] {row['feature']:<22} PSI = {row['psi']:.4f}")
    alerts = int((report["status"] == "ALERT").sum())

    if alerts == 0:
        print("\nNo feature in ALERT: the champion keeps serving. Nothing to retrain.")
        return

    print(f"\nSTEP 2 - {alerts} feature(s) in ALERT: training a new candidate and running the gate")
    decision = run_gate("retrain", trigger="psi_drift", drift_alerts=alerts)
    print(f"\nWORKFLOW RESULT: {decision}")


if __name__ == "__main__":
    main()