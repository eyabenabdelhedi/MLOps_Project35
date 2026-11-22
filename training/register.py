import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

EXPERIMENT = "churn-scoring"
MODEL_NAME = "churn-model"
# Only tree models can be explained by shap.TreeExplainer (required by the brief).
TREE_MODELS = {"random_forest", "gradient_boosting"}


def alias_version(alias):
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except Exception:
        return None


def register(run):
    """Register a run's model, reusing the existing version if there is one."""
    try:
        for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if mv.run_id == run.info.run_id:
                return mv.version
    except Exception:
        pass
    return mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME).version


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
if not ranked:
    raise SystemExit("No finished tree-model runs. Run training/train.py first.")

champion = alias_version("champion")

if champion is None:
    # FIRST TIME ONLY: bootstrap the champion from the best validation AUC.
    if len(ranked) < 2:
        raise SystemExit("Need finished runs for both tree models. Run training/train.py first.")
    best, second = ranked[0], ranked[1]
    v_champ, v_chall = register(best), register(second)
    client.set_registered_model_alias(MODEL_NAME, "champion", v_champ)
    client.set_registered_model_alias(MODEL_NAME, "challenger", v_chall)
    print(f"BOOTSTRAP @champion   -> v{v_champ} {best.data.params['model_type']} "
          f"val AUC={best.data.metrics['val_roc_auc']:.3f}")
    print(f"          @challenger -> v{v_chall} {second.data.params['model_type']} "
          f"val AUC={second.data.metrics['val_roc_auc']:.3f}")
else:
    # The champion is NEVER touched here: only training/promote.py may move it.
    # The best candidate that is not the champion becomes @challenger.
    for r in ranked:
        v = register(r)
        if str(v) != str(champion):
            client.set_registered_model_alias(MODEL_NAME, "challenger", v)
            print(f"@champion unchanged -> v{champion}")
            print(f"@challenger -> v{v} {r.data.params['model_type']} "
                  f"val AUC={r.data.metrics['val_roc_auc']:.3f}")
            break
    else:
        print(f"No candidate other than the champion (v{champion}). Nothing to do.")
    print("Now run: python training/promote.py")