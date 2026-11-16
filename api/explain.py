"""
SHAP explanations for the churn model.

explain_customer() takes the scikit-learn pipeline loaded from MLflow and
ONE customer (a one-row DataFrame). It returns:
  - the churn probability,
  - the SHAP contribution of every feature,
  - a waterfall plot (PNG, base64),
  - a plain-text narrative built only from those SHAP numbers.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")          # draw without a window (needed inside a web server)
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "monthly_usage_hours",
    "support_tickets",
    "payment_delays",
]

# Creating a TreeExplainer takes time, so we keep one per model version
_explainer_cache = {"version": None, "explainer": None}


def _get_explainer(pipeline, version):
    if _explainer_cache["version"] != version:
        tree_model = pipeline.named_steps["clf"]
        _explainer_cache["explainer"] = shap.TreeExplainer(tree_model)
        _explainer_cache["version"] = version
    return _explainer_cache["explainer"]


def _clean_name(name):
    # "num__tenure_months" -> "tenure_months", "cat__contract_type_monthly" -> "contract_type_monthly"
    return name.split("__", 1)[-1]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def explain_customer(pipeline, customer_df, version="unknown", top_n=3):
    pre = pipeline.named_steps["preprocess"]
    tree_model = pipeline.named_steps["clf"]
    explainer = _get_explainer(pipeline, version)

    # 1. Preprocess exactly like in training, then ask SHAP
    X = pre.transform(customer_df)
    feature_names = [_clean_name(n) for n in pre.get_feature_names_out()]

    shap_values = explainer.shap_values(X)

    # Different SHAP/model combinations return different shapes: keep class 1 = "leaves"
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    values = shap_values[0]

    base = np.atleast_1d(explainer.expected_value)
    base_value = float(base[1] if len(base) > 1 else base[0])

    # 2. Which scale are the numbers on?
    #    Random Forest -> probability points; Gradient Boosting -> log-odds.
    on_probability_scale = isinstance(tree_model, RandomForestClassifier)
    unit = "probability points" if on_probability_scale else "log-odds units"

    raw_output = base_value + float(values.sum())
    shap_probability = raw_output if on_probability_scale else float(_sigmoid(raw_output))
    base_probability = base_value if on_probability_scale else float(_sigmoid(base_value))

    # Honesty check: SHAP must reproduce the model's real probability
    model_probability = float(pipeline.predict_proba(customer_df)[0][1])
    additivity_gap = abs(shap_probability - model_probability)

    # 3. Real (human) values to display, instead of scaled numbers
    display_values = []
    for i, name in enumerate(feature_names):
        if name in NUMERIC_COLUMNS:
            display_values.append(float(customer_df[name].iloc[0]))
        else:
            display_values.append(float(X[0][i]))      # one-hot: 1 = yes, 0 = no

    # 4. Contributions table, biggest effect first
    contributions = [
        {
            "feature": name,
            "value": display_values[i],
            "shap_value": round(float(values[i]), 5),
        }
        for i, name in enumerate(feature_names)
    ]
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    # 5. Waterfall plot
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=np.array(display_values),
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    buffer = io.BytesIO()
    plt.gcf().savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close("all")
    waterfall_png = base64.b64encode(buffer.getvalue()).decode("ascii")

    # 6. Plain-text narrative: ONLY numbers computed above
    up = [c for c in contributions if c["shap_value"] > 0][:top_n]
    down = [c for c in contributions if c["shap_value"] < 0][:top_n]

    def describe(c):
        return f"{c['feature']} = {c['value']:g} ({c['shap_value']:+.3f})"

    lines = [
        f"The model estimates a {model_probability:.1%} probability that this customer "
        f"will leave, compared with an average of {base_probability:.1%} for the "
        f"training customers."
    ]
    if up:
        lines.append("Factors raising the risk: " + "; ".join(describe(c) for c in up) + ".")
    if down:
        lines.append("Factors lowering the risk: " + "; ".join(describe(c) for c in down) + ".")
    lines.append(f"Contributions are expressed in {unit}.")
    narrative = " ".join(lines)

    return {
        "churn_probability": round(model_probability, 4),
        "base_probability": round(base_probability, 4),
        "unit": unit,
        "additivity_gap": round(additivity_gap, 6),
        "contributions": contributions,
        "narrative": narrative,
        "waterfall_png_base64": waterfall_png,
    }