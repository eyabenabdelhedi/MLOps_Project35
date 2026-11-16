import numpy as np
import pandas as pd

rng = np.random.default_rng(42)   # 42 = fixed seed, so everyone gets the same "random" data
n = 20000                         # number of customers

# 1) Invent the customers' characteristics
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n).astype(float),         # how long they've been a customer
    "monthly_charges": rng.normal(120, 40, n).clip(20, 300),       # what they pay per month
    "monthly_usage_hours": rng.normal(80, 30, n).clip(0, 250),     # how much they use the service
    "support_tickets": rng.poisson(2, n).astype(float),            # number of complaints
    "payment_delays": rng.poisson(1, n).astype(float),             # times they paid late
    "contract_type": rng.choice(["monthly", "annual", "biennial"], n, p=[0.5, 0.3, 0.2]),
    "company_size": rng.choice(["small", "medium", "large"], n, p=[0.5, 0.35, 0.15]),
})

# 2) A hidden "rule" that decides who is likely to leave
logit = (
    -0.5
    - 0.035 * df.tenure_months                           # long-time customers leave less
    + 0.012 * (df.monthly_charges - 120)                 # expensive -> leave more
    - 0.015 * (df.monthly_usage_hours - 80)              # heavy users leave less
    + 0.35 * df.support_tickets                          # many complaints -> leave more
    + 0.6 * df.payment_delays                            # late payers -> leave more
    + np.where(df.contract_type == "monthly", 1.0,
               np.where(df.contract_type == "annual", -0.3, -0.8))
    + np.where(df.company_size == "small", 0.3, 0.0)
)
prob = 1 / (1 + np.exp(-logit))                          # turn the score into a probability (0 to 1)

# 3) Roll a dice for each customer using that probability: 1 = left, 0 = stayed
df["churn"] = (rng.random(n) < prob).astype(int)

df.to_csv("data/customers.csv", index=False)
print(df.shape, "churn rate:", round(df.churn.mean(), 3))