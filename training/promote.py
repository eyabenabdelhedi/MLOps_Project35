"""Promotion gate: the @challenger replaces @champion ONLY if it beats it
on the frozen holdout (F1 AND ROC-AUC). Every decision is logged.

  python training/promote.py                  # normal run
  python training/promote.py --min-delta 0.005
  python training/promote.py --rollback       # put @previous back as @champion
"""
import argparse
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "churn-model"
HOLDOUT = ROOT / "data" / "holdout.csv"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def version_of(alias):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except Exception:
        return None


def evaluate(version, X, y):
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    proba = model.predict_proba(X)[:, 1]
    return {"auc": float(roc_auc_score(y, proba)),
            "f1": float(f1_score(y, (proba >= 0.5).astype(int)))}


def log_decision(rec):
    """Save the decision in MLflow and in the Django database."""
    mlflow.set_experiment("promotion-gate")
    with mlflow.start_run(run_name=f"{rec['decision']}_v{rec['challenger_version']}"):
        mlflow.log_params({k: rec[k] for k in
                           ("champion_version", "challenger_version", "decision", "min_delta")})
        mlflow.log_metrics({k: rec[k] for k in
                            ("champion_f1", "champion_auc", "challenger_f1", "challenger_auc")})
        mlflow.set_tag("reason", rec["reason"])
    try:
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        import django
        django.setup()
        from api.models import PromotionLog
        PromotionLog.objects.create(**rec)
    except Exception as exc:
        print(f"(warning) could not write PromotionLog in Django: {exc}")


def rollback():
    prev, champ = version_of("previous"), version_of("champion")
    if prev is None:
        raise SystemExit("No @previous alias: nothing to roll back to.")
    client.set_registered_model_alias(MODEL_NAME, "champion", prev)
    client.set_registered_model_alias(MODEL_NAME, "previous", champ)
    log_decision({"champion_version": str(champ), "challenger_version": str(prev),
                  "champion_f1": 0.0, "champion_auc": 0.0,
                  "challenger_f1": 0.0, "challenger_auc": 0.0, "min_delta": 0.0,
                  "decision": "rolled_back",
                  "reason": f"Manual rollback: @champion v{champ} -> v{prev}"})
    print(f"ROLLED BACK: @champion is now v{prev} (v{champ} saved as @previous)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-delta", type=float, default=0.002,
                    help="challenger must beat the champion by more than this on F1 AND AUC")
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()

    if args.rollback:
        return rollback()

    champ, chall = version_of("champion"), version_of("challenger")
    if champ is None or chall is None:
        raise SystemExit("Need both @champion and @challenger. Run training/register.py first.")
    if champ == chall:
        raise SystemExit(f"@champion and @challenger are the same version (v{champ}). Nothing to compare.")

    holdout = pd.read_csv(HOLDOUT)
    y = holdout["churn"]
    X = holdout.drop(columns=[c for c in ("churn", "source_row") if c in holdout.columns])

    c, h = evaluate(champ, X, y), evaluate(chall, X, y)
    beats_auc = h["auc"] > c["auc"] + args.min_delta
    beats_f1 = h["f1"] > c["f1"] + args.min_delta

    print(f"Holdout ({len(holdout)} rows, never used for training or model selection)")
    print(f"  champion   v{champ}: AUC={c['auc']:.4f}  F1={c['f1']:.4f}")
    print(f"  challenger v{chall}: AUC={h['auc']:.4f}  F1={h['f1']:.4f}")

    if beats_auc and beats_f1:
        decision = "promoted"
        reason = (f"Challenger beat the champion by more than {args.min_delta} on both "
                  f"AUC ({h['auc']:.4f} vs {c['auc']:.4f}) and F1 ({h['f1']:.4f} vs {c['f1']:.4f}).")
        client.set_registered_model_alias(MODEL_NAME, "previous", champ)
        client.set_registered_model_alias(MODEL_NAME, "champion", chall)
    else:
        decision = "kept"
        failed = [n for n, ok in (("AUC", beats_auc), ("F1", beats_f1)) if not ok]
        reason = (f"Challenger did not beat the champion by more than {args.min_delta} "
                  f"on: {', '.join(failed)}. Champion kept.")

    log_decision({"champion_version": str(champ), "challenger_version": str(chall),
                  "champion_f1": c["f1"], "champion_auc": c["auc"],
                  "challenger_f1": h["f1"], "challenger_auc": h["auc"],
                  "min_delta": args.min_delta, "decision": decision, "reason": reason})

    print(f"\nDECISION: {decision.upper()}")
    print(reason)
    if decision == "promoted":
        print(f"@champion -> v{chall}   @previous -> v{champ}")


if __name__ == "__main__":
    main()
    