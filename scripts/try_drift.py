import json

import pandas as pd
import requests

URL = "http://127.0.0.1:8000/api/drift/check/"
df = pd.read_csv("data/holdout.csv").drop(columns="churn")


def send(batch, label):
    recs = json.loads(batch.to_json(orient="records"))   # NaN -> null
    r = requests.post(URL, json={"records": recs}).json()
    print(label, "->", r["status"], "| max PSI =", r["max_psi"])
    for k, v in r["features"].items():
        print(f"   {k:22s} {v['psi']:.3f}  {v['level']}")


send(df, "Normal batch")

shifted = df.copy()
shifted["monthly_charges"] *= 1.6
shifted["support_tickets"] += 3
send(shifted, "Shifted batch")
