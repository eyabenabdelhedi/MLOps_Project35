import sys
sys.path.insert(0, "training")
from workflow import load_pool

batch = load_pool().sample(400, random_state=1).copy()
batch["monthly_charges"] *= 1.6
batch["support_tickets"] += 3
batch.to_csv("data/new_batch.csv", index=False)
print("data/new_batch.csv written:", len(batch), "rows")
