import sys
sys.path.insert(0, ".")

from api.narrative import build_facts, verify

result = {
    "churn_probability": 0.423, "base_probability": 0.300, "unit": "probability points",
    "contributions": [
        {"feature": "support_tickets", "value": 5.0, "shap_value": 0.12345},
        {"feature": "tenure_months", "value": 8.0, "shap_value": -0.04321},
    ],
}
facts = build_facts(result)

good = ("The model estimates 42.3% churn risk against an average of 30.0%. "
        "support_tickets = 5 (+0.123) raises the risk; tenure_months = 8 (-0.043) lowers it. "
        "Values are in probability points.")

tests = [
    ("good", good),
    ("wrong shap number", good.replace("+0.123", "+0.150")),
    ("unknown feature", good.replace("tenure_months", "contract_type")),
    ("invented figure", good + " About 70% of such customers leave."),
]

for name, text in tests:
    ok, problems = verify(text, facts)
    print(f"{name:18s} accepted={ok}  {problems}")