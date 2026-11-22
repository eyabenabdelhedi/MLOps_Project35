"""Drift -> retrain -> gate workflow.

  python training/workflow.py --simulate stable            # no drift: nothing happens
  python training/workflow.py --simulate drifted           # drift: retrain + gate
  python training/workflow.py --batch data/my_batch.csv    # a real production batch
  python training/workflow.py --force                      # retrain regardless of drift
  python training/workflow.py --batch b.csv --append-labelled new_labelled.csv
  python training/workflow.py --simulate drifted --dry-run # report only, run nothing
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from monitoring.psi_batch import psi_report  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "churn-model"
NUM = ["tenure_months", "monthly_charges", "monthly_usage_hours",
       "support_tickets", "payment_delays"]
CAT = ["contract_type", "company_size"]

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def champion_reference():
    """PSI reference logged with the CURRENT champion (fallback: data/reference.csv)."""
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    try:
        path = download_artifacts(run_id=mv.run_id, artifact_path="reference.csv")
    except Exception as exc:
        print(f"(warning) no reference artifact for champion v{mv.version} ({exc}); "
              "using data/reference.csv")
        path = ROOT / "data" / "reference.csv"
    return pd.read_csv(path), mv.version


def simulate(ref, kind):
    """Build a production batch from the reference, stable or deliberately drifted."""
    rng = np.random.default_rng(2026)
    b = ref.sample(1000, replace=True, random_state=1).reset_index(drop=True)
    if kind == "drifted":
        b["monthly_charges"] = b["monthly_charges"] * 1.4
        b["tenure_months"] = (b["tenure_months"] * 0.6).round()
        b["support_tickets"] = b["support_tickets"] + rng.poisson(2, len(b))
        switch = rng.random(len(b)) < 0.5
        b.loc[switch, "contract_type"] = "monthly"
    return b


def log_drift(report):
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        import django
        django.setup()
        from api.models import DriftCheckLog
        DriftCheckLog.objects.create(n_rows=report["n_rows"], max_psi=report["max_psi"],
                                     status=report["status"], features=report["features"])
    except Exception as exc:
        print(f"(warning) could not write DriftCheckLog in Django: {exc}")


def append_labelled(path):
    """Append new LABELLED customers to the END of customers.csv.

    Appending (never inserting) keeps the frozen holdout rows intact.
    The file is renamed afterwards so the same rows are never appended twice.
    """
    path = Path(path)
    new = pd.read_csv(path)
    cust_path = ROOT / "data" / "customers.csv"
    cust = pd.read_csv(cust_path)
    missing = set(cust.columns) - set(new.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    pd.concat([cust, new[cust.columns]], ignore_index=True).to_csv(cust_path, index=False)
    path.rename(path.with_name(path.name + ".appended"))
    print(f"Appended {len(new)} labelled rows to data/customers.csv "
          f"(now {len(cust) + len(new)} rows).")


def run(script):
    print(f"\n>>> python training/{script}")
    subprocess.run([sys.executable, str(ROOT / "training" / script)],
                   cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--batch", help="CSV with a production batch (feature columns)")
    src.add_argument("--simulate", choices=["stable", "drifted"])
    ap.add_argument("--force", action="store_true", help="retrain even without drift")
    ap.add_argument("--retrain-on", choices=["alert", "warning"], default="alert",
                    help="lowest drift status that triggers retraining")
    ap.add_argument("--append-labelled", help="CSV of new labelled customers to add first")
    ap.add_argument("--dry-run", action="store_true", help="report only, run nothing")
    args = ap.parse_args()

    ref, champ_version = champion_reference()
    print(f"Champion: v{champ_version}")

    status = "forced"
    if not args.force:
        if args.simulate:
            batch = simulate(ref, args.simulate)
        elif args.batch:
            batch = pd.read_csv(args.batch)
        else:
            raise SystemExit("Give --batch, --simulate or --force.")
        if len(batch) < 30:
            raise SystemExit("A batch needs at least 30 customers for a meaningful PSI.")

        report = psi_report(ref, batch, NUM, CAT)
        status = report["status"]
        print(f"\nPSI on {report['n_rows']} customers vs the champion's training data:")
        for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
            print(f"  {name:<22} PSI={f['psi']:.4f}  {f['level']}")
        print(f"Overall status: {status.upper()} (max PSI {report['max_psi']:.4f})")
        if not args.dry_run:
            log_drift(report)

    trigger = {"alert"} if args.retrain_on == "alert" else {"alert", "warning"}
    if not args.force and status not in trigger:
        print(f"\nNo retraining needed (status '{status}').")
        return
    if args.dry_run:
        print(f"\nDRY RUN: would retrain now (status '{status}').")
        return

    print(f"\nRetraining triggered (status '{status}').")
    if args.append_labelled:
        append_labelled(args.append_labelled)
    run("train.py")
    run("register.py")
    run("promote.py")

    champ = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    print(f"\nWorkflow finished. @champion is now v{champ}.")


if __name__ == "__main__":
    main()