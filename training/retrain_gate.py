"""
Verification gate: a challenger becomes @champion ONLY if it beats the
current champion on BOTH F1 and ROC-AUC, measured on the SAME holdout set.

Usage (run from the project root):
    python training/retrain_gate.py --mode registered   # compare @challenger vs @champion
    python training/retrain_gate.py --mode weak         # train a deliberately weak candidate, then compare
    python training/retrain_gate.py --mode retrain      # train a new, tuned candidate, then compare

Other scripts (e.g. training/workflow.py) can call run_gate() directly.
"""

import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MLFLOW_URI = "http://localhost:5000"
MODEL_NAME = "CustomerChurn"
THRESHOLD = 0.5

NUM = ["tenure_months", "monthly_charges", "monthly_usage_hours",
       "support_tickets", "payment_delays"]
CAT = ["contract_type", "company_size"]

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

# The two kinds of candidate this script can train and register as @challenger
CANDIDATES = {
    "weak": {
        "run_name": "weak_candidate",
        "model_type": "weak_gradient_boosting",
        "make": lambda: GradientBoostingClassifier(
            n_estimators=15, learning_rate=0.05, max_depth=1, random_state=42),
    },
    "retrain": {
        "run_name": "retrained_candidate",
        "model_type": "retrained_gradient_boosting",
        "make": lambda: GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=7),
    },
}


# ---------- 1) The exam: SAME split as train.py (same seed, same stratification) ----------
def load_split():
    df = pd.read_csv("data/customers.csv")
    X, y = df.drop(columns="churn"), df["churn"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ---------- 2) Grading one model ----------
def evaluate(pipeline, X_test, y_test):
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {"auc": roc_auc_score(y_test, proba), "f1": f1_score(y_test, pred)}


def version_of(alias):
    return client.get_model_version_by_alias(MODEL_NAME, alias).version


def load_version(version):
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


# ---------- 3) Training a candidate and pointing @challenger at it ----------
def build_pipeline(clf):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("preprocess", pre), ("clf", clf)])


def register_candidate(kind, X_train, y_train):
    """Train a candidate ('weak' or 'retrain'), register it, and point @challenger at it."""
    spec = CANDIDATES[kind]
    clf = spec["make"]()
    mlflow.set_experiment("churn-scoring")
    pipe = build_pipeline(clf).fit(X_train, y_train)
    with mlflow.start_run(run_name=spec["run_name"]):
        mlflow.log_param("model_type", spec["model_type"])
        mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()})
        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(X_train, pipe.predict_proba(X_train)),
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
            registered_model_name=MODEL_NAME,
        )
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"Registered a '{kind}' candidate as version {version} and set it as @challenger.")


# ---------- 4) The gate ----------
def run_gate(candidate=None, trigger="manual", drift_alerts=None):
    """
    Compare @challenger with @champion and promote only if it wins on AUC and F1.

    candidate    : None (use the existing @challenger), 'weak' or 'retrain' (train a new one first)
    trigger      : 'manual' or 'psi_drift' (who started this run; saved in MLflow)
    drift_alerts : number of PSI alerts that triggered the run (saved in MLflow)

    Returns "PROMOTED", "REJECTED" or "SKIPPED".
    """
    X_train, X_test, y_train, y_test = load_split()

    if candidate:
        register_candidate(candidate, X_train, y_train)

    champion_version = version_of("champion")
    challenger_version = version_of("challenger")

    if champion_version == challenger_version:
        print("Champion and challenger are the same version: nothing to compare.")
        return "SKIPPED"

    champion = evaluate(load_version(champion_version), X_test, y_test)
    challenger = evaluate(load_version(challenger_version), X_test, y_test)

    better_auc = challenger["auc"] > champion["auc"]
    better_f1 = challenger["f1"] > champion["f1"]
    promote = better_auc and better_f1

    print(f"\nSame holdout set: {len(X_test)} customers")
    print(f"{'':22s}{'AUC':>10s}{'F1':>10s}")
    print(f"{'Champion  (v' + str(champion_version) + ')':22s}{champion['auc']:>10.4f}{champion['f1']:>10.4f}")
    print(f"{'Challenger (v' + str(challenger_version) + ')':22s}{challenger['auc']:>10.4f}{challenger['f1']:>10.4f}")
    print(f"Challenger beats champion on AUC: {better_auc}")
    print(f"Challenger beats champion on F1 : {better_f1}")

    # Record the decision in MLflow (visible in the UI, useful for the slides)
    mlflow.set_experiment("promotion-gate")
    with mlflow.start_run(run_name=f"gate_v{challenger_version}_vs_v{champion_version}"):
        mlflow.log_params({"champion_version": champion_version,
                           "challenger_version": challenger_version})
        mlflow.log_metrics({"champion_auc": champion["auc"], "champion_f1": champion["f1"],
                            "challenger_auc": challenger["auc"], "challenger_f1": challenger["f1"]})
        if drift_alerts is not None:
            mlflow.log_metric("drift_alert_count", drift_alerts)
        mlflow.set_tag("trigger", trigger)
        mlflow.set_tag("decision", "PROMOTED" if promote else "REJECTED")

    if promote:
        client.set_registered_model_alias(MODEL_NAME, "previous", champion_version)  # keep a way back
        client.set_registered_model_alias(MODEL_NAME, "champion", challenger_version)
        print(f"\nDECISION: PROMOTED. @champion now points to version {challenger_version} "
              f"(old champion v{champion_version} kept as @previous).")
        return "PROMOTED"

    print(f"\nDECISION: REJECTED. @champion stays on version {champion_version}.")
    return "REJECTED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["registered", "weak", "retrain"], default="registered")
    args = parser.parse_args()
    run_gate(None if args.mode == "registered" else args.mode)


if __name__ == "__main__":
    main()