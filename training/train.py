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

# 1) Load the data and split it: 80% study book (train), 20% exam (test)
df = pd.read_csv("data/customers.csv")
X, y = df.drop(columns="churn"), df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Snapshot of the training data, used later by the PSI drift monitor
Path("data").mkdir(exist_ok=True)
X_train.to_csv("data/reference.csv", index=False)

NUM = ["tenure_months", "monthly_charges", "monthly_usage_hours",
       "support_tickets", "payment_delays"]
CAT = ["contract_type", "company_size"]


# 2) The conveyor belt: preprocessing + model in ONE object
def build_pipeline(clf):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("preprocess", pre), ("clf", clf)])


# 3) The three candidate machines
candidates = {
    "logreg": LogisticRegression(max_iter=1000, C=1.0),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                    max_depth=3, random_state=42),
}

# 4) Train each one and write everything in the MLflow notebook
for name, clf in candidates.items():
    with mlflow.start_run(run_name=name):
        pipe = build_pipeline(clf).fit(X_train, y_train)   # learns from the study book only
        proba = pipe.predict_proba(X_test)[:, 1]            # probability of churn on the exam
        pred = (proba >= 0.5).astype(int)

        mlflow.log_param("model_type", name)
        mlflow.log_params({f"clf__{k}": v for k, v in clf.get_params().items()})
        mlflow.log_metrics({
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, pred),
        })

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        RocCurveDisplay.from_predictions(y_test, proba, ax=ax[0])
        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax[1])
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
        print(f"{name:18s} AUC={roc_auc_score(y_test, proba):.3f}  "
              f"F1={f1_score(y_test, pred):.3f}")