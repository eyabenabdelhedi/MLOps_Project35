"""
1. Builds two production batches:
     - a STABLE one (resampled from the training data)
     - a DRIFTED one (we change customer behaviour on purpose)
2. Runs the PSI inspector on both and prints the lights per feature.
3. Saves a chart and a CSV report for the slides.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psi import ALERT_THRESHOLD, WARNING_THRESHOLD, psi_report

DATA = Path("data")
OUT = Path("monitoring")
BATCH_SIZE = 1000

reference = pd.read_csv(DATA / "reference.csv")
rng = np.random.default_rng(2026)

# --- Batch 1: stable (same kind of customers as in training) ---
stable = reference.sample(BATCH_SIZE, replace=True, random_state=1).reset_index(drop=True)

# --- Batch 2: drifted (we change 5 features on purpose, 2 stay untouched) ---
drifted = reference.sample(BATCH_SIZE, replace=True, random_state=2).reset_index(drop=True)
drifted["monthly_charges"] = drifted["monthly_charges"] * 1.4              # prices went up
drifted["tenure_months"] = (drifted["tenure_months"] * 0.6).round()        # newer customers
drifted["support_tickets"] = drifted["support_tickets"] + rng.poisson(2, BATCH_SIZE)
drifted["payment_delays"] = drifted["payment_delays"] + rng.binomial(1, 0.5, BATCH_SIZE)
switch = rng.random(BATCH_SIZE) < 0.5                                       # half move to monthly
drifted.loc[switch, "contract_type"] = "monthly"
# monthly_usage_hours and company_size are NOT changed: they should stay green

stable.to_csv(DATA / "production_stable.csv", index=False)
drifted.to_csv(DATA / "production_drifted.csv", index=False)

# --- Inspect ---
icon = {"STABLE": "OK     ", "WARNING": "WARNING", "ALERT": "ALERT  "}

for title, batch in [("STABLE batch", stable), ("DRIFTED batch", drifted)]:
    report = psi_report(reference, batch)
    print(f"\n=== PSI per feature: {title} ({len(batch)} customers vs training) ===")
    for _, row in report.iterrows():
        print(f"  [{icon[row['status']]}] {row['feature']:<22} PSI = {row['psi']:.4f}")
    n_alerts = int((report["status"] == "ALERT").sum())
    print(f"  -> {n_alerts} feature(s) in ALERT")
    report.to_csv(OUT / f"psi_report_{title.split()[0].lower()}.csv", index=False)

# --- Chart for the slides (drifted batch) ---
report = psi_report(reference, drifted).iloc[::-1]
colors = report["status"].map({"STABLE": "#2e9e5b", "WARNING": "#e0a800", "ALERT": "#d64545"})
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh(report["feature"], report["psi"], color=colors)
ax.axvline(WARNING_THRESHOLD, linestyle="--", color="gray", linewidth=1)
ax.axvline(ALERT_THRESHOLD, linestyle="--", color="black", linewidth=1)
ax.set_xlabel("PSI (0.10 = warning, 0.25 = alert)")
ax.set_title("PSI drift per feature: drifted production batch")
fig.tight_layout()
fig.savefig(OUT / "psi_drifted.png", dpi=150)
print("\nSaved: monitoring/psi_drifted.png, two CSV reports, two batch CSV files in data/")