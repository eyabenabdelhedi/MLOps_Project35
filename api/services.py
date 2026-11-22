import logging
import threading
import time
from mlflow.artifacts import download_artifacts
import mlflow
import mlflow.sklearn
import pandas as pd
from django.conf import settings
from mlflow import MlflowClient
from .narrative import build_narrative

from .explain import explain_customer

log = logging.getLogger(__name__)

MODEL_NAME = getattr(settings, "MODEL_NAME", "churn-model")
TRACKING_URI = getattr(settings, "MLFLOW_TRACKING_URI", "http://localhost:5000")
ALIAS = "champion"
ALIAS_TTL_SECONDS = 30   # how often we ask MLflow whether @champion moved

NUM = ["tenure_months", "monthly_charges", "monthly_usage_hours",
       "support_tickets", "payment_delays"]
CAT = ["contract_type", "company_size"]
COLUMNS = NUM + CAT

_cache = {"version": None, "run_id": None, "pipe": None, "checked_at": 0.0}
_model_lock = threading.Lock()   # protects the model cache
_plot_lock = threading.Lock()    # matplotlib / SHAP plotting is not thread-safe


def get_champion():
    """Return (pipeline, version) for the current @champion.

    - Asks MLflow which version @champion points to at most every 30 seconds.
    - Loads the model by its exact version number, so the label always matches.
    - If MLflow is unreachable but a model is already cached, keeps serving it.
    """
    with _model_lock:
        now = time.time()
        if _cache["pipe"] is not None and now - _cache["checked_at"] < ALIAS_TTL_SECONDS:
            return _cache["pipe"], _cache["version"]

        mlflow.set_tracking_uri(TRACKING_URI)
        try:
            mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
        except Exception as exc:
            if _cache["pipe"] is not None:
                log.warning("MLflow unreachable (%s); serving cached v%s",
                            exc, _cache["version"])
                return _cache["pipe"], _cache["version"]
            raise

        if _cache["version"] != mv.version:   # first call, or alias moved
            pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{mv.version}")
            clf = pipe.named_steps["clf"]
            if not hasattr(clf, "estimators_"):   # not a tree ensemble
                raise RuntimeError(
                    f"The @champion model ({type(clf).__name__}) is not a tree model, "
                    "so shap.TreeExplainer cannot explain it.")
            _cache.update(version=mv.version, run_id=mv.run_id, pipe=pipe)
            log.info("Loaded %s v%s as @%s", MODEL_NAME, mv.version, ALIAS)

        _cache["checked_at"] = now
        return _cache["pipe"], _cache["version"]


def to_frame(payload: dict) -> pd.DataFrame:
    """One customer (dict) -> DataFrame with the training columns, missing ones as NaN."""
    return pd.DataFrame([payload]).reindex(columns=COLUMNS)


def predict(payload: dict) -> dict:
    pipe, version = get_champion()
    proba = float(pipe.predict_proba(to_frame(payload))[0, 1])
    return {"churn_probability": round(proba, 4),
            "prediction": int(proba >= 0.5),
            "model_version": version}


def explain(payload: dict, top_k: int = 5) -> dict:
    pipe, version = get_champion()
    df = to_frame(payload)

    with _plot_lock:
        r = explain_customer(pipe, df, version=str(version), top_n=3)

    if r["additivity_gap"] > 1e-3:
        log.warning("SHAP additivity gap %.6f for model v%s",
                    r["additivity_gap"], version)

    drivers = [
        {**c, "direction": "increases churn risk" if c["shap_value"] > 0
         else "decreases churn risk"}
        for c in r["contributions"][:top_k]
    ]

    return {
        "churn_probability": r["churn_probability"],
        "model_version": version,
        "base_probability": r["base_probability"],
        "unit": r["unit"],
        "additivity_gap": r["additivity_gap"],
        "top_drivers": drivers,
        "narrative": r["narrative"],
        "waterfall_png_base64": r["waterfall_png_base64"],
    }
def get_reference():
    """PSI reference of the CURRENT champion (logged as an artifact at training time).

    Falls back to data/reference.csv for models trained before this change.
    """
    get_champion()   # makes sure the cache is fresh
    try:
        path = download_artifacts(run_id=_cache["run_id"], artifact_path="reference.csv")
        return pd.read_csv(path)
    except Exception as exc:
        log.warning("No reference artifact for run %s (%s); using data/reference.csv",
                    _cache["run_id"], exc)
        return pd.read_csv(settings.BASE_DIR / "data" / "reference.csv")