import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, roc_auc_score,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-scoring")

DATA = Path("data")
DATA.mkdir(exist_ok=True)
HOLDOUT = DATA / "holdout.csv"
REFERENCE = DATA / "reference.csv"

NUM = ["tenure_months", "monthly_charges", "monthly_usage_hours",
       "support_tickets", "payment_delays"]
CAT = ["contract_type", "company_size"]


def freeze_holdout(df):
    """Create the holdout ONCE. Later runs reuse it and never overwrite it.

    Each holdout row keeps 'source_row' = its row number in customers.csv, so
    future training can exclude exactly those rows (new rows must be APPENDED
    to customers.csv, never inserted in the middle).
    """
    if HOLDOUT.exists():
        h = pd.read_csv(HOLDOUT)
        if "source_row" in h.columns:
            return h
        print("Old holdout.csv has no 'source_row': rebuilding it (same rows).")
    _, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["churn"], random_state=42)
    h = df.loc[test_idx].copy()
    h.insert(0, "source_row", h.index)
    h.to_csv(HOLDOUT, index=False)
    print(f"Holdout frozen: {len(h)} rows -> {HOLDOUT}")
    return h


# 1) Load the data, exclude the frozen holdout, split the rest 75/25 train/validation
df = pd.read_csv("data/customers.csv")
holdout = freeze_holdout(df)
pool = df.drop(index=holdout["source_row"], errors="ignore")

X, y = pool.drop(columns="churn"), pool["churn"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# Reference sample for the PSI drift monitor (also logged inside every MLflow run)
X_train.to_csv(REFERENCE, index=False)


# 2) Preprocessing + model in ONE Pipeline (nothing is learned outside the training rows)
def build_pipeline(clf):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("preprocess", pre), ("clf", clf)])


# 3) The three candidates
candidates = {
    "logreg": LogisticRegression(max_iter=1000, C=1.0),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                    max_depth=3, random_state=42),
}

# 4) Train each one, evaluate on VALIDATION, log everything to MLflow
for name, clf in candidates.items():
    with mlflow.start_run(run_name=name):
        pipe = build_pipeline(clf).fit(X_train, y_train)
        proba = pipe.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(int)

        val_auc = roc_auc_score(y_val, proba)
        val_pr = average_precision_score(y_val, proba)
        val_f1 = f1_score(y_val, pred)

        mlflow.log_param("model_type", name)
        mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()})
        mlflow.log_params({"n_train": len(X_train), "n_val": len(X_val),
                           "n_holdout": len(holdout)})
        mlflow.log_metrics({
            "val_roc_auc": val_auc,
            "val_pr_auc": val_pr,
            "val_f1": val_f1,
        })

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        RocCurveDisplay.from_predictions(y_val, proba, ax=ax[0])
        PrecisionRecallDisplay.from_predictions(y_val, proba, ax=ax[1])
        mlflow.log_figure(fig, "curves.png")
        plt.close(fig)

        # The PSI reference travels with the model: drift is measured against
        # the data THIS model was trained on.
        mlflow.log_artifact(str(REFERENCE))

        signature = infer_signature(X_train, pipe.predict_proba(X_train))
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        print(f"{name:18s} val AUC={val_auc:.3f}  PR-AUC={val_pr:.3f}  F1={val_f1:.3f}")