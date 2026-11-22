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

# 1) Load the data and split it three ways: 60% train, 20% validation, 20% test
df = pd.read_csv("data/customers.csv")
X, y = df.drop(columns="churn"), df["churn"]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)  # 25% of 80% = 20%

Path("data").mkdir(exist_ok=True)
# Reference data for the PSI drift monitor
X_train.to_csv("data/reference.csv", index=False)
# Locked holdout: saved here, never used in this script (only the Step 4 gate reads it)
X_test.assign(churn=y_test).to_csv("data/holdout.csv", index=False)

NUM = ["tenure_months", "monthly_charges", "monthly_usage_hours",
       "support_tickets", "payment_delays"]
CAT = ["contract_type", "company_size"]


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

        signature = infer_signature(X_train, pipe.predict_proba(X_train))
        mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        print(f"{name:18s} val AUC={val_auc:.3f}  PR-AUC={val_pr:.3f}  F1={val_f1:.3f}")