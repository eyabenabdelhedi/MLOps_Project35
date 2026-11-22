import base64
import io

import matplotlib
matplotlib.use("Agg")  # no window, server-safe
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from django.conf import settings
from mlflow import MlflowClient

MODEL_NAME = getattr(settings, "MODEL_NAME", "churn-model")
TRACKING_URI = getattr(settings, "MLFLOW_TRACKING_URI", "http://localhost:5000")
ALIAS = "champion"

NUM = ["tenure_months", "monthly_charges", "monthly_usage_hours",
       "support_tickets", "payment_delays"]
CAT = ["contract_type", "company_size"]
COLUMNS = NUM + CAT

_cache = {"version": None, "pipe": None, "explainer": None, "names": None}


def get_champion():
    """Return (pipeline, explainer, feature_names, version) for the current @champion."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)

    if _cache["version"] != mv.version:  # first call, or alias moved to a new version
        pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
        clf = pipe.named_steps["clf"]
        try:
            explainer = shap.TreeExplainer(clf)
        except Exception as exc:
            raise RuntimeError(
                f"The @champion model ({type(clf).__name__}) is not a tree model, "
                "so shap.TreeExplainer cannot explain it.") from exc
        names = [n.split("__", 1)[-1]
                 for n in pipe.named_steps["preprocess"].get_feature_names_out()]
        _cache.update(version=mv.version, pipe=pipe, explainer=explainer, names=names)

    return _cache["pipe"], _cache["explainer"], _cache["names"], _cache["version"]


def to_frame(payload: dict) -> pd.DataFrame:
    """One customer (dict) -> DataFrame with the training columns, missing ones as NaN."""
    return pd.DataFrame([payload]).reindex(columns=COLUMNS)


def predict(payload: dict) -> dict:
    pipe, _, _, version = get_champion()
    proba = float(pipe.predict_proba(to_frame(payload))[0, 1])
    return {"churn_probability": round(proba, 4),
            "prediction": int(proba >= 0.5),
            "model_version": version}


def _positive_class(shap_values, expected_value):
    """Normalise SHAP output across model types to (values[n, features], base_value)."""
    if isinstance(shap_values, list):            # older shap: one array per class
        return np.asarray(shap_values[1]), float(np.ravel(expected_value)[1])
    sv = np.asarray(shap_values)
    if sv.ndim == 3:                             # newer shap: (n, features, classes)
        return sv[:, :, 1], float(np.ravel(expected_value)[1])
    return sv, float(np.ravel(expected_value)[0])  # gradient boosting: (n, features)


def explain(payload: dict, top_k: int = 5) -> dict:
    pipe, explainer, names, version = get_champion()
    df = to_frame(payload)
    proba = float(pipe.predict_proba(df)[0, 1])

    X_t = pipe.named_steps["preprocess"].transform(df)
    X_t = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
    sv, base = _positive_class(explainer.shap_values(X_t), explainer.expected_value)

    order = np.argsort(-np.abs(sv[0]))[:top_k]
    drivers = [{"feature": names[i],
                "value": float(X_t[0, i]),
                "shap_value": float(sv[0, i]),
                "direction": "increases churn risk" if sv[0, i] > 0 else "decreases churn risk"}
               for i in order]

    exp = shap.Explanation(values=sv[0], base_values=base, data=X_t[0], feature_names=names)
    shap.plots.waterfall(exp, max_display=10, show=False)
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close("all")

    return {"churn_probability": round(proba, 4),
            "model_version": version,
            "base_value": base,
            "top_drivers": drivers,
            "waterfall_png_base64": base64.b64encode(buf.getvalue()).decode()}