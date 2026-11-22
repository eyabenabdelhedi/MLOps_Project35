import numpy as np
import pandas as pd

EPS = 1e-4  # avoids log(0) when a bin is empty


def level(v: float) -> str:
    if v < 0.1:
        return "stable"
    if v < 0.25:
        return "moderate"
    return "significant"


def _psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    cur = pd.to_numeric(cur, errors="coerce").dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:          # nearly constant feature
        return 0.0
    inner = edges[1:-1]
    e = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1) / len(ref)
    a = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(inner) + 1) / len(cur)
    return _psi(e, a)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    e = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    a = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return _psi(e, a)


def psi_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame, num_cols, cat_cols) -> dict:
    features = {}
    for c in num_cols:
        v = psi_numeric(ref_df[c], cur_df[c]) if c in cur_df else 0.0
        features[c] = {"psi": round(v, 4), "level": level(v)}
    for c in cat_cols:
        v = psi_categorical(ref_df[c], cur_df[c]) if c in cur_df else 0.0
        features[c] = {"psi": round(v, 4), "level": level(v)}

    max_psi = max(f["psi"] for f in features.values())
    return {"n_rows": len(cur_df),
            "max_psi": max_psi,
            "status": {"stable": "ok", "moderate": "warning", "significant": "alert"}[level(max_psi)],
            "features": features}