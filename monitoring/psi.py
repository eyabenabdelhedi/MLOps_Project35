"""
PSI (Population Stability Index): how different is a new batch of customers
from the customers the model was trained on?
"""

import numpy as np
import pandas as pd

EPSILON = 1e-4            # avoids log(0) and division by zero when a box is empty
WARNING_THRESHOLD = 0.10
ALERT_THRESHOLD = 0.25


def _psi_from_shares(expected, actual):
    """The PSI formula: sum of (actual - expected) * ln(actual / expected)."""
    expected = np.clip(np.asarray(expected, dtype=float), EPSILON, None)
    actual = np.clip(np.asarray(actual, dtype=float), EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_numeric(reference, current, bins=10):
    """For continuous numbers (e.g. monthly_charges): 10 boxes with equal
    numbers of training customers in each."""
    reference = pd.Series(reference).dropna().to_numpy(dtype=float)
    current = pd.Series(current).dropna().to_numpy(dtype=float)

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf       # new values may fall outside the old range

    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    return _psi_from_shares(expected, actual)


def psi_categorical(reference, current):
    """For words or few distinct values (e.g. contract_type, payment_delays):
    each distinct value is its own box."""
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    expected = reference.value_counts(normalize=True)
    actual = current.value_counts(normalize=True)

    boxes = expected.index.union(actual.index)  # includes values never seen in training
    expected = expected.reindex(boxes, fill_value=0.0)
    actual = actual.reindex(boxes, fill_value=0.0)
    return _psi_from_shares(expected.to_numpy(), actual.to_numpy())


def status_from_psi(psi):
    if psi >= ALERT_THRESHOLD:
        return "ALERT"
    if psi >= WARNING_THRESHOLD:
        return "WARNING"
    return "STABLE"


def psi_feature(reference, current):
    """Pick the right method for one column. Returns (psi, method)."""
    reference = pd.Series(reference)
    is_number = pd.api.types.is_numeric_dtype(reference)

    if is_number and reference.nunique() > 10:
        return psi_numeric(reference, current), "numeric (10 quantile boxes)"

    if is_number:                               # few distinct numbers, e.g. 0, 1, 2, 3
        return (psi_categorical(reference.astype(float), pd.Series(current).astype(float)),
                "discrete values")

    return psi_categorical(reference, current), "categories"


def psi_report(reference_df, current_df):
    """One row per feature, biggest drift first."""
    rows = []
    for column in reference_df.columns:
        value, method = psi_feature(reference_df[column], current_df[column])
        rows.append({
            "feature": column,
            "psi": round(value, 4),
            "status": status_from_psi(value),
            "method": method,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)