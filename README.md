# MLOps Project #35: End-to-End Customer Churn Prediction

An end-to-end MLOps prototype for B2B customer churn: leak-free training with MLflow, a model registry with a `@champion` alias, a Django REST scoring service with per-customer SHAP explanations, PSI drift detection, and an automated retrain-and-promote workflow guarded by a verification gate.

**Stack:** Python, scikit-learn, MLflow, SHAP, Django REST Framework, PostgreSQL, pandas, Pipenv.

## How the project maps to the four steps

| Step | What it does | Main files |
|---|---|---|
| 1. Leak-free pipeline and MLflow logging | Trains 3 candidates (logistic regression, random forest, gradient boosting), each wrapped in one `Pipeline` (imputation, scaling, encoding, model). Logs hyperparameters, ROC-AUC, PR-AUC, F1, ROC/PR curves and the input/output signature. | `training/train.py` |
| 2. Registry promotion and `@champion` serving | Registers the best **tree** model by validation ROC-AUC as `@champion` (runner-up as `@challenger`). Django loads `models:/churn-model@champion` dynamically and builds a `shap.TreeExplainer`. | `training/register.py`, `api/services.py` |
| 3. SHAP UI and PSI drift detector | A web page renders the SHAP waterfall plot for one customer. A PSI detector scores incoming production batches. | `api/ui.py`, `api/drift.py`, `monitoring/` |
| 4. Workflow automation and verification gate | A drift alert triggers retraining. The new model becomes `@champion` only if it beats the current champion on **both** F1 and ROC-AUC on a locked holdout. The explanation text is checked against the exact SHAP numbers. | `training/workflow.py`, `training/promote.py`, `api/narrative.py` |

### Data splits (this is what keeps the evaluation honest)

- **Train (60%)**: models learn from it. Preprocessing is fitted inside the `Pipeline`, so nothing leaks.
- **Validation (20%)**: used by `register.py` to pick the champion.
- **Holdout (20%)**: saved to `data/holdout.csv`, never used in training or model selection. Only the promotion gate reads it.

## Project structure

```
api/          Django app: scoring, explanation and drift endpoints, SHAP page, narrative, models
config/       Django configuration (settings.py, urls.py)
data/         customers.csv, reference.csv (PSI reference), holdout.csv (locked test set)
monitoring/   PSI drift module
training/     train.py, register.py, promote.py (gate), workflow.py
scripts/      small helper scripts to try the API
docs/         example SHAP waterfall images
Pipfile       dependencies and shortcut commands (Pipfile.lock pins exact versions)
.env.example  template for the PostgreSQL settings
manage.py     Django entry point
```

`data/customers.csv` is a synthetic B2B churn dataset.

## Installation (Windows, PowerShell)

```powershell
git clone https://github.com/eyabenabdelhedi/MLOps_Project35.git
cd MLOps_Project35
py -m pip install --user pipenv
py -m pipenv install --dev
```

`pipenv install` reads `Pipfile` and `Pipfile.lock`, so everyone gets exactly the same package versions.

Copy `.env.example` to `.env` and put your PostgreSQL password in it (`.env` is ignored by Git). Pipenv loads this file automatically.

Every command below is run as `py -m pipenv run <command>`. Shortcuts defined in the `Pipfile`: `mlflow-server`, `train`, `register`, `gate`, `workflow`, `migrate`, `serve`.

## PostgreSQL

Django stores its tables and the audit logs (`api_predictionlog`, `api_driftchecklog`, `api_promotionlog`) in PostgreSQL when `USE_POSTGRES=1`. Without it, Django falls back to SQLite.

1. Create an empty database named `mlops_churn`.
2. Set these values in `.env` (see `.env.example`): `USE_POSTGRES=1`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`.
3. Create the tables: `py -m pipenv run migrate`

The password is read from the environment and is never stored in the code. MLflow keeps its own data in `mlflow.db` and `mlartifacts/`, separate from Django's database.

## Running the project

Use three terminals, all in the project folder.

**Terminal 1: MLflow tracking server and registry** (leave it running)

```powershell
py -m pipenv run mlflow-server
```

MLflow UI: http://127.0.0.1:5000

**Terminal 2: train, register, then start Django**

```powershell
py -m pipenv run train
py -m pipenv run register
py -m pipenv run migrate
py -m pipenv run serve
```

- `train.py` creates 3 runs in the `churn-scoring` experiment and writes `data/reference.csv` and `data/holdout.csv`.
- `register.py` ranks only the tree models by validation ROC-AUC. The first time, it sets the best as `@champion` and the runner-up as `@challenger`. Once a champion exists, it only refreshes `@challenger`. A new `@champion` can only come from the gate.

**Using the app**

- SHAP explanation page: http://127.0.0.1:8000/api/ui/
- Fill in a customer and click "Explain this customer" to get the churn probability, the waterfall plot, the top drivers with exact SHAP values, and a short explanation text.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/predict/` | POST | Churn probability for one customer |
| `/api/explain/` | POST | Probability, top SHAP drivers, waterfall image (base64), narrative |
| `/api/drift/check/` | POST | PSI per feature for a batch of customers |
| `/api/ui/` | GET | Web page for per-customer SHAP explanations |

Customer fields: `tenure_months`, `monthly_charges`, `monthly_usage_hours`, `support_tickets`, `payment_delays`, `contract_type`, `company_size`.

Example:

```powershell
$body = '{"tenure_months": 8, "monthly_charges": 120.5, "monthly_usage_hours": 40, "support_tickets": 5, "payment_delays": 2, "contract_type": "monthly", "company_size": "small"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/explain/ -Method Post -ContentType "application/json" -Body $body
```

## Drift detection and automated retraining

PSI (Population Stability Index) compares a production batch with the training data of the current champion (stored as an artifact of its MLflow run):

| PSI | Level |
|---|---|
| below 0.1 | stable |
| 0.1 to 0.25 | moderate (warning) |
| 0.25 or more | significant (alert) |

**Run the workflow** (MLflow running):

```powershell
py -m pipenv run workflow --simulate drifted --dry-run    # report only, nothing is retrained
py -m pipenv run workflow --simulate drifted              # drift alert -> retrain -> register -> gate
py -m pipenv run workflow --batch data\new_batch.csv      # use your own batch of customers
py -m pipenv run workflow --force                         # retrain without a drift alert
```

Other options: `--retrain-on warning` (also react to moderate drift) and `--append-labelled FILE.csv` (add new labelled customers to the end of `customers.csv` first, which keeps the holdout rows unchanged).

What `workflow.py` does:

1. Computes PSI for the batch against the champion's reference data and logs it in `api_driftchecklog`.
2. If the status reaches the alert level, it runs `train.py`, then `register.py`.
3. Runs the **verification gate** (`training/promote.py`): the `@challenger` is compared with the `@champion` on `data/holdout.csv`. It is promoted only if it beats the champion by more than `min_delta` (default 0.002) on **both** F1 and ROC-AUC. The old champion is kept as `@previous`. Every decision is saved in `api_promotionlog` and in MLflow.

Run the gate alone with `py -m pipenv run gate`, or undo a promotion with `py -m pipenv run gate --rollback`.

## Explanations from exact SHAP values

`api/narrative.py` builds the explanation text from the exact numbers returned by `shap.TreeExplainer`. With an `ANTHROPIC_API_KEY` set, an LLM may write the text, but it is accepted only if every number and every feature name in it appears in the SHAP facts and the churn probability is stated. Otherwise (no key, API error, or failed check) the page shows the template text built directly from the SHAP values. SHAP values are in the model's own output units (log-odds for gradient boosting, probability points for random forest).

## Tests

```powershell
py -m pipenv run python manage.py test api
```

The tests check the narrative verification: exact formatting of the facts, acceptance of a faithful text, and rejection of an invented number, an unknown feature, or a missing probability.

## Example results

| Model | Validation ROC-AUC |
|---|---|
| Logistic regression | 0.816 (baseline, not served) |
| Gradient boosting | 0.796 |
| Random forest | 0.792 |

Logistic regression scores slightly higher but cannot be explained with `shap.TreeExplainer`, so only tree models compete for `@champion`. The versions currently holding `@champion` and `@challenger` are shown in the MLflow model registry.

Drift workflow on a simulated shifted batch (1000 customers): PSI flagged `tenure_months` (2.471), `support_tickets` (1.192) and `monthly_charges` (0.987) as significant and `contract_type` (0.232) as moderate, while `monthly_usage_hours`, `company_size` and `payment_delays` stayed stable. The alert triggered retraining. On the 1200-row holdout, the new challenger (v7) scored AUC 0.7972 and F1 0.7156, identical to the champion (v3), because the simulated batch carries no new labels. The gate therefore **kept the champion**: a challenger must beat it by more than 0.002 on both AUC and F1. A real promotion needs new labelled customers (`--append-labelled`).

Example SHAP waterfall: `docs/waterfall_api.png`

## Limitations

- The holdout is a single fixed split, so repeated gate runs reuse the same data. A production system would refresh it.
- The drift endpoint only reports PSI; retraining is started by running `training/workflow.py`, not by the endpoint itself.
- The dataset is synthetic.
- Models are saved with cloudpickle, which should only be loaded from trusted sources (here, your own MLflow server).
## Option B: Run with Docker (one command)

Requires Docker Desktop. This starts PostgreSQL, MLflow and the Django website already connected, with no Python, Pipenv or PostgreSQL install needed.

```bash
git clone https://github.com/eyabenabdelhedi/MLOps_Project35.git
cd MLOps_Project35
docker compose up --build -d
docker compose ps
```

Wait until `postgres` and `mlflow` show `healthy` and `web` shows `Up`. The first build takes a few minutes.

Then train and register the models inside the web container:

```bash
docker compose exec web python training/train.py
docker compose exec web python training/register.py
```

Open:

- MLflow: http://127.0.0.1:5000
- SHAP explanation page: http://127.0.0.1:8000/api/ui/

Other commands:

```bash
docker compose exec web python training/workflow.py --simulate drifted --dry-run
docker compose exec web python training/workflow.py --simulate drifted
docker compose exec web python training/promote.py
docker compose exec web python training/promote.py --rollback
docker compose exec web python manage.py test api
docker compose logs web
docker compose down        # stop (add -v to also delete stored data)
```

Notes:

- Ports 5000 and 8000 must be free (stop any local MLflow or Django first).
- The database password defaults to `postgres`. To change it, set `POSTGRES_PASSWORD` in a local `.env` file (never commit it).
- Docker has its own empty MLflow registry, so run `train` and `register` the first time.
- If you see file errors on Windows, move the project out of OneDrive (for example `C:\dev\MLOps_Project35`).

## Contributors

- Eya Ibn Abdelhedi
- Midani Amira
- Araar Ghofrane
- Yengui Amine

## Academic project

DS1, Subject #35: End-to-End MLOps Lifecycle with MLflow, Model Registry and SHAP. ISG Tunis, supervised by Chaouki Bayoudhi.