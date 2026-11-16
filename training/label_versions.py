"""
Adds readable labels (tags + description) to every version of CustomerChurn.
The algorithm name is read from the training run itself (the 'model_type'
parameter that train.py logged), so nothing is typed by hand.
"""

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

MODEL_NAME = "CustomerChurn"

NICE_NAMES = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "weak_gradient_boosting": "Weak Gradient Boosting (gate demo)",
    "logreg": "Logistic Regression",
}

for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"),
                 key=lambda v: int(v.version)):

    run = client.get_run(mv.run_id)
    model_type = run.data.params.get("model_type", "unknown")
    nice = NICE_NAMES.get(model_type, model_type)

    client.set_model_version_tag(MODEL_NAME, mv.version, "algorithm", nice)
    client.update_model_version(
        MODEL_NAME, mv.version,
        description=f"{nice} - trained in run {mv.run_id[:8]}"
    )
    print(f"version {mv.version}: {nice}")

print("\nCurrent aliases:")
for alias in ["champion", "challenger", "previous"]:
    try:
        v = client.get_model_version_by_alias(MODEL_NAME, alias).version
        print(f"  @{alias} -> version {v}")
    except Exception:
        print(f"  @{alias} -> (not set)")