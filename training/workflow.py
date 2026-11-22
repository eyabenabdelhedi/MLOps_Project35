import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier

from gate import MODEL_NAME, run_gate
from monitoring.psi_batch import psi_report
from pipeline import CAT, NUM, build_pipeline

mlflow.set_tracking_uri("http://localhost:5000")
PSI_THRESHOLD = 0.25   # "significant" drift


def load_pool() -> pd.DataFrame:
    """All labelled customers EXCEPT the locked holdout rows."""
    full = pd.read_csv("data/customers.csv")
    hold = pd.read_csv("data/holdout.csv").drop_duplicates()
    merged = full.merge(hold, how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge")


def retrain_candidate(batch: pd.DataFrame) -> str:
    pool = load_pool()
    train = pd.concat([pool, batch], ignore_index=True) if "churn" in batch.columns else pool
    X, y = train.drop(columns="churn")[NUM + CAT], train["churn"]

    clf = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                     max_depth=3, random_state=42)
    pipe = build_pipeline(clf).fit(X, y)

    mlflow.set_experiment("churn-scoring")
    with mlflow.start_run(run_name="retrain_after_drift"):
        mlflow.log_param("model_type", "gradient_boosting_retrain")  # ignored by register.py
        mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()})
        mlflow.log_metric("train_rows", len(train))
        info = mlflow.sklearn.log_model(
            pipe, name="model",
            signature=infer_signature(X, pipe.predict_proba(X)),
            input_example=X.head(3),
            serialization_format="cloudpickle")
    mv = mlflow.register_model(info.model_uri, MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "challenger", mv.version)
    return mv.version


def main(batch_path: str):
    batch = pd.read_csv(batch_path)
    ref = pd.read_csv("data/reference.csv")
    report = psi_report(ref, batch, NUM, CAT)
    print(f"Drift check on {report['n_rows']} rows: max PSI={report['max_psi']} -> {report['status']}")
    for k, v in report["features"].items():
        print(f"   {k:22s} {v['psi']:.3f}  {v['level']}")

    if report["max_psi"] < PSI_THRESHOLD:
        print("No significant drift: no retraining.")
        return
    print("Significant drift: retraining a candidate...")
    v = retrain_candidate(batch)
    print(f"Candidate registered as @challenger (v{v}). Running the gate...")
    run_gate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python training/workflow.py data/new_batch.csv")
    main(sys.argv[1])
