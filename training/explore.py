import pandas as pd

df = pd.read_csv("data/customers.csv")

print(df.head())
print()
print("Rows, columns:", df.shape)
print("Churn rate:", round(df["churn"].mean(), 3))
print()
print(df.dtypes)