import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score

mlflow.set_tracking_uri("http://localhost:5000")
MODEL_NAME = "churn-model"


def _evaluate(alias, X, y):
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
    proba = model.predict_proba(X)[:, 1]
    return {"f1": float(f1_score(y, (proba >= 0.5).astype(int))),
            "auc": float(roc_auc_score(y, proba))}


def run_gate(min_delta: float = 0.0) -> dict:
    client = MlflowClient()
    try:
        champ_v = client.get_model_version_by_alias(MODEL_NAME, "champion").version
        chall_v = client.get_model_version_by_alias(MODEL_NAME, "challenger").version
    except Exception:
        print("Gate: no @challenger to evaluate.")
        return {"promoted": False, "reason": "no challenger"}
    if champ_v == chall_v:
        print("Gate: champion and challenger are the same version.")
        return {"promoted": False, "reason": "same version"}

    hold = pd.read_csv("data/holdout.csv")
    X, y = hold.drop(columns="churn"), hold["churn"]
    champ = _evaluate("champion", X, y)
    chall = _evaluate("challenger", X, y)

    promoted = (chall["f1"] > champ["f1"] + min_delta
                and chall["auc"] > champ["auc"] + min_delta)

    mlflow.set_experiment("promotion-gate")
    with mlflow.start_run(run_name=f"gate_v{chall_v}_vs_v{champ_v}"):
        mlflow.log_params({"champion_version": champ_v, "challenger_version": chall_v,
                           "min_delta": min_delta, "holdout_rows": len(hold)})
        mlflow.log_metrics({"champion_f1": champ["f1"], "champion_auc": champ["auc"],
                            "challenger_f1": chall["f1"], "challenger_auc": chall["auc"],
                            "promoted": int(promoted)})

    print(f"Champion   v{champ_v}: F1={champ['f1']:.4f}  AUC={champ['auc']:.4f}")
    print(f"Challenger v{chall_v}: F1={chall['f1']:.4f}  AUC={chall['auc']:.4f}")

    if promoted:
        client.set_registered_model_alias(MODEL_NAME, "previous", champ_v)
        client.set_registered_model_alias(MODEL_NAME, "champion", chall_v)
        client.delete_registered_model_alias(MODEL_NAME, "challenger")
        print(f"PROMOTED: v{chall_v} is the new @champion (old one kept as @previous).")
    else:
        print("NOT promoted: the challenger must beat the champion on BOTH F1 and AUC.")
    return {"promoted": promoted, "champion": champ, "challenger": chall}


if __name__ == "__main__":
    run_gate()