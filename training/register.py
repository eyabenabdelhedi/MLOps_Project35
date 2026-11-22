import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

EXPERIMENT = "churn-scoring"
MODEL_NAME = "churn-model"
# Only tree models can be explained by shap.TreeExplainer (required by the brief).
# logreg stays in MLflow as a baseline but is never served.
TREE_MODELS = {"random_forest", "gradient_boosting"}

exp = client.get_experiment_by_name(EXPERIMENT)
runs = client.search_runs(
    [exp.experiment_id],
    filter_string="metrics.val_roc_auc > 0",
    order_by=["attributes.start_time DESC"],
)

# Keep only the most recent run of each tree model type
latest = {}
for r in runs:
    mt = r.data.params["model_type"]
    if mt in TREE_MODELS:
        latest.setdefault(mt, r)

ranked = sorted(latest.values(),
                key=lambda r: r.data.metrics["val_roc_auc"], reverse=True)
if len(ranked) < 2:
    raise SystemExit("Need finished runs for both tree models. Run training/train.py first.")


def register(run):
    """Register a run's model, reusing the existing version if there is one."""
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if mv.run_id == run.info.run_id:
            return mv.version
    return mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME).version


best, second = ranked[0], ranked[1]
v_champ = register(best)
v_chall = register(second)

client.set_registered_model_alias(MODEL_NAME, "champion", v_champ)
client.set_registered_model_alias(MODEL_NAME, "challenger", v_chall)

print(f"@champion   -> v{v_champ}  {best.data.params['model_type']}  "
      f"val AUC={best.data.metrics['val_roc_auc']:.3f}")
print(f"@challenger -> v{v_chall}  {second.data.params['model_type']}  "
      f"val AUC={second.data.metrics['val_roc_auc']:.3f}")