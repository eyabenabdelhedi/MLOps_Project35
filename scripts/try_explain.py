import mlflow
import mlflow.sklearn
import pandas as pd
import base64

from api.explain import explain_customer

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.sklearn.load_model("models:/CustomerChurn@champion")

customer = pd.DataFrame([{
    "tenure_months": 8.0,
    "monthly_charges": 120.5,
    "monthly_usage_hours": 40.0,
    "support_tickets": 5.0,
    "payment_delays": 2.0,
    "contract_type": "monthly",
    "company_size": "small",
}])

result = explain_customer(model, customer, version="test")

print(result["narrative"])
print("\nAdditivity gap (should be ~0):", result["additivity_gap"])
print("Unit:", result["unit"])

with open("waterfall_test.png", "wb") as f:
    f.write(base64.b64decode(result["waterfall_png_base64"]))
print("Saved waterfall_test.png")