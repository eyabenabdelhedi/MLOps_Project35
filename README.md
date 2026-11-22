# MLOps Project #35: End-to-End Customer Churn Prediction

An end-to-end MLOps prototype for B2B customer churn: leak-free training with MLflow, a model registry with a `@champion` alias, a Django REST scoring service with per-customer SHAP explanations, PSI drift detection, and an automated retrain-and-promote workflow guarded by a verification gate.

**Stack:** Python, scikit-learn, MLflow, SHAP, Django REST Framework, PostgreSQL (optional), pandas.

## How the project maps to the four steps

| Step | What it does | Main files |
|---|---|---|
| 1. Leak-free pipeline and MLflow logging | Trains 3 candidates (logistic regression, random forest, gradient boosting), each wrapped in one `Pipeline` (imputation, scaling, encoding, model). Logs hyperparameters, ROC-AUC, PR-AUC, F1, ROC/PR curves and the input/output signature. | `training/train.py`, `training/pipeline.py` |
| 2. Registry promotion and `@champion` serving | Registers the best **tree** model by validation ROC-AUC as `@champion` (runner-up as `@challenger`). Django loads `models:/churn-model@champion` dynamically and builds a `shap.TreeExplainer`. | `training/register.py`, `api/services.py` |
| 3. SHAP UI and PSI drift detector | A web page renders the SHAP waterfall plot for one customer. A PSI detector scores incoming production batches. | `api/ui.py`, `monitoring/psi_batch.py`, `api/views.py` |
| 4. Workflow automation and verification gate | A drift alert triggers retraining of a real candidate. The candidate becomes `@champion` only if it beats the current champion on **both** F1 and ROC-AUC on a locked holdout. The explanation text is generated only from exact SHAP values. | `training/workflow.py`, `training/gate.py`, `api/narrative.py` |

### Data splits (this is what keeps the evaluation honest)

- **Train (60%)**: models learn from it. Preprocessing is fitted inside the `Pipeline`, so nothing leaks.
- **Validation (20%)**: used by `register.py` to pick the champion.
- **Holdout (20%)**: saved to `data/holdout.csv`, never used in training or model selection. Only the promotion gate reads it.

## Project structure

```
api/          Django app: scoring and explanation endpoints, SHAP page, narrative, models
config/       Django configuration (settings.py, urls.py)
data/         customers.csv, reference.csv (PSI reference), holdout.csv (locked test set)
monitoring/   PSI drift module (psi_batch.py)
training/     train.py, register.py, pipeline.py, workflow.py, gate.py
scripts/      small helper scripts to try the API and the workflow
docs/         example SHAP waterfall images
manage.py     Django entry point
```

`data/customers.csv` is a synthetic B2B churn dataset (see `training/generate_data.py`).

## Installation (Windows, PowerShell)

```powershell
git clone https://github.com/eyabenabdelhedi/MLOps_Project35.git
cd MLOps_Project35
pipenv install -r requirements.txt
```

Every command below is run through `pipenv run ...` (or inside `pipenv shell`).

## Running the project

Use three terminals, all in the project folder.

**Terminal 1: MLflow tracking server and registry** (leave it running)

```powershell
pipenv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

MLflow UI: http://127.0.0.1:5000

**Terminal 2: train, register, then start Django**

```powershell
pipenv run python training/train.py
pipenv run python training/register.py
pipenv run python manage.py migrate
pipenv run python manage.py runserver
```

- `train.py` creates 3 runs in the `churn-scoring` experiment and writes `data/reference.csv` and `data/holdout.csv`.
- `register.py` registers the best tree model as `@champion` and the other as `@challenger`.
- Run `train.py` and `register.py` once at the start. **Do not run `register.py` again after the gate has promoted a model**, because it re-ranks by validation AUC and would overwrite the gate's decision.

**Using the app**

- SHAP explanation page: http://127.0.0.1:8000/api/ui/
- Fill in a customer and click "Explain this customer" to get the churn probability, the waterfall plot, the top drivers with exact SHAP values, and a short explanation text.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/predict/` | POST | Churn probability for one customer |
| `/api/explain/` | POST | Probability, top SHAP drivers, waterfall image (base64), narrative |
| `/api/drift/check/` | POST | PSI per feature for a batch (`{"records": [...]}`, at least 30 rows) |
| `/api/ui/` | GET | Web page for per-customer SHAP explanations |

Customer fields: `tenure_months`, `monthly_charges`, `monthly_usage_hours`, `support_tickets`, `payment_delays`, `contract_type`, `company_size`.

Example:

```powershell
$body = '{"tenure_months": 12, "monthly_charges": 80, "monthly_usage_hours": 40, "support_tickets": 3, "payment_delays": 1, "contract_type": "<value from customers.csv>", "company_size": "<value from customers.csv>"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/explain/ -Method Post -ContentType "application/json" -Body $body
```

## Drift detection and automated retraining

PSI (Population Stability Index) compares a production batch with the training data in `data/reference.csv`:

| PSI | Level |
|---|---|
| below 0.1 | stable |
| 0.1 to 0.25 | moderate |
| 0.25 or more | significant (alert) |

**Try the drift endpoint** (Django and MLflow running):

```powershell
pipenv run python scripts/try_drift.py
```

**Run the full workflow** on a batch CSV:

```powershell
pipenv run python scripts/try_workflow.py                      # creates data/new_batch.csv with shifted values
pipenv run python training/workflow.py data/new_batch.csv
```

What `workflow.py` does:

1. Computes PSI for the batch against the reference data.
2. If the maximum PSI is 0.25 or more, retrains a gradient boosting candidate on all non-holdout data (plus the batch, when it has a `churn` column) and registers it as `@challenger`.
3. Runs the **verification gate** (`training/gate.py`): the challenger is evaluated against the champion on `data/holdout.csv`. It is promoted only if it beats the champion on **both** F1 and ROC-AUC. On promotion, the old champion is kept as `@previous`. Every gate decision is logged to the `promotion-gate` experiment in MLflow.

## Explanations from exact SHAP values

`api/narrative.py` builds the explanation text from a fixed template filled with the exact numbers returned by `shap.TreeExplainer`. A check function verifies that every number in the text comes from the SHAP output, so the text cannot contain an invented figure. SHAP values are in the model's own output units (log-odds for gradient boosting).

## PostgreSQL (optional)

By default Django uses SQLite and needs no configuration. To use PostgreSQL for Django's tables and for the audit log of predictions (`api_predictionlog`) and drift checks (`api_driftchecklog`):

```powershell
$env:USE_POSTGRES = "1"
$env:POSTGRES_PASSWORD = "your_password"
# optional: POSTGRES_DB (default mlops_churn), POSTGRES_USER (postgres), POSTGRES_HOST (localhost), POSTGRES_PORT (5432)
pipenv run python manage.py migrate
```

Create an empty database named `mlops_churn` first. The password is read from the environment and is never stored in the code. MLflow keeps its own data in `mlflow.db` and `mlartifacts/`, separate from Django's database.

## Example results

| Model | Validation ROC-AUC |
|---|---|
| Logistic regression | 0.816 (baseline, not served) |
| Gradient boosting | 0.796 (`@champion`) |
| Random forest | 0.792 (`@challenger`) |

Logistic regression scores slightly higher but cannot be explained with `shap.TreeExplainer`, so only tree models compete for `@champion`.

Drift workflow on a shifted test batch: PSI flagged `monthly_charges` (1.807) and `support_tickets` (6.642) as significant while the other features stayed stable. The retrained candidate scored F1 0.7154 and AUC 0.7935 on the holdout against F1 0.7179 and AUC 0.7963 for the champion, so the gate **did not promote** it.

Example SHAP waterfall: `docs/waterfall_api.png`

## Limitations

- The holdout is a single fixed split, so repeated gate runs reuse the same data. A production system would refresh it.
- The drift endpoint reports PSI; retraining is started by running `training/workflow.py`, not by the endpoint itself.
- The dataset is synthetic.
- Models are saved with cloudpickle, which should only be loaded from trusted sources (here, your own MLflow server).

## Academic project

DS1, Subject #35: End-to-End MLOps Lifecycle with MLflow, Model Registry and SHAP.# MLOps Project #35: End-to-End Customer Churn Prediction

An end-to-end MLOps prototype for B2B customer churn: leak-free training with MLflow, a model registry with a `@champion` alias, a Django REST scoring service with per-customer SHAP explanations, PSI drift detection, and an automated retrain-and-promote workflow guarded by a verification gate.

**Stack:** Python, scikit-learn, MLflow, SHAP, Django REST Framework, PostgreSQL (optional), pandas.

## How the project maps to the four steps

| Step | What it does | Main files |
|---|---|---|
| 1. Leak-free pipeline and MLflow logging | Trains 3 candidates (logistic regression, random forest, gradient boosting), each wrapped in one `Pipeline` (imputation, scaling, encoding, model). Logs hyperparameters, ROC-AUC, PR-AUC, F1, ROC/PR curves and the input/output signature. | `training/train.py`, `training/pipeline.py` |
| 2. Registry promotion and `@champion` serving | Registers the best **tree** model by validation ROC-AUC as `@champion` (runner-up as `@challenger`). Django loads `models:/churn-model@champion` dynamically and builds a `shap.TreeExplainer`. | `training/register.py`, `api/services.py` |
| 3. SHAP UI and PSI drift detector | A web page renders the SHAP waterfall plot for one customer. A PSI detector scores incoming production batches. | `api/ui.py`, `monitoring/psi_batch.py`, `api/views.py` |
| 4. Workflow automation and verification gate | A drift alert triggers retraining of a real candidate. The candidate becomes `@champion` only if it beats the current champion on **both** F1 and ROC-AUC on a locked holdout. The explanation text is generated only from exact SHAP values. | `training/workflow.py`, `training/gate.py`, `api/narrative.py` |

### Data splits (this is what keeps the evaluation honest)

- **Train (60%)**: models learn from it. Preprocessing is fitted inside the `Pipeline`, so nothing leaks.
- **Validation (20%)**: used by `register.py` to pick the champion.
- **Holdout (20%)**: saved to `data/holdout.csv`, never used in training or model selection. Only the promotion gate reads it.

## Project structure

```
api/          Django app: scoring and explanation endpoints, SHAP page, narrative, models
config/       Django configuration (settings.py, urls.py)
data/         customers.csv, reference.csv (PSI reference), holdout.csv (locked test set)
monitoring/   PSI drift module (psi_batch.py)
training/     train.py, register.py, pipeline.py, workflow.py, gate.py
scripts/      small helper scripts to try the API and the workflow
docs/         example SHAP waterfall images
manage.py     Django entry point
```

`data/customers.csv` is a synthetic B2B churn dataset (see `training/generate_data.py`).

## Installation (Windows, PowerShell)

```powershell
git clone https://github.com/eyabenabdelhedi/MLOps_Project35.git
cd MLOps_Project35
pipenv install -r requirements.txt
```

Every command below is run through `pipenv run ...` (or inside `pipenv shell`).

## Running the project

Use three terminals, all in the project folder.

**Terminal 1: MLflow tracking server and registry** (leave it running)

```powershell
pipenv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

MLflow UI: http://127.0.0.1:5000

**Terminal 2: train, register, then start Django**

```powershell
pipenv run python training/train.py
pipenv run python training/register.py
pipenv run python manage.py migrate
pipenv run python manage.py runserver
```

- `train.py` creates 3 runs in the `churn-scoring` experiment and writes `data/reference.csv` and `data/holdout.csv`.
- `register.py` registers the best tree model as `@champion` and the other as `@challenger`.
- Run `train.py` and `register.py` once at the start. **Do not run `register.py` again after the gate has promoted a model**, because it re-ranks by validation AUC and would overwrite the gate's decision.

**Using the app**

- SHAP explanation page: http://127.0.0.1:8000/api/ui/
- Fill in a customer and click "Explain this customer" to get the churn probability, the waterfall plot, the top drivers with exact SHAP values, and a short explanation text.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/predict/` | POST | Churn probability for one customer |
| `/api/explain/` | POST | Probability, top SHAP drivers, waterfall image (base64), narrative |
| `/api/drift/check/` | POST | PSI per feature for a batch (`{"records": [...]}`, at least 30 rows) |
| `/api/ui/` | GET | Web page for per-customer SHAP explanations |

Customer fields: `tenure_months`, `monthly_charges`, `monthly_usage_hours`, `support_tickets`, `payment_delays`, `contract_type`, `company_size`.

Example:

```powershell
$body = '{"tenure_months": 12, "monthly_charges": 80, "monthly_usage_hours": 40, "support_tickets": 3, "payment_delays": 1, "contract_type": "<value from customers.csv>", "company_size": "<value from customers.csv>"}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/explain/ -Method Post -ContentType "application/json" -Body $body
```

## Drift detection and automated retraining

PSI (Population Stability Index) compares a production batch with the training data in `data/reference.csv`:

| PSI | Level |
|---|---|
| below 0.1 | stable |
| 0.1 to 0.25 | moderate |
| 0.25 or more | significant (alert) |

**Try the drift endpoint** (Django and MLflow running):

```powershell
pipenv run python scripts/try_drift.py
```

**Run the full workflow** on a batch CSV:

```powershell
pipenv run python scripts/try_workflow.py                      # creates data/new_batch.csv with shifted values
pipenv run python training/workflow.py data/new_batch.csv
```

What `workflow.py` does:

1. Computes PSI for the batch against the reference data.
2. If the maximum PSI is 0.25 or more, retrains a gradient boosting candidate on all non-holdout data (plus the batch, when it has a `churn` column) and registers it as `@challenger`.
3. Runs the **verification gate** (`training/gate.py`): the challenger is evaluated against the champion on `data/holdout.csv`. It is promoted only if it beats the champion on **both** F1 and ROC-AUC. On promotion, the old champion is kept as `@previous`. Every gate decision is logged to the `promotion-gate` experiment in MLflow.

## Explanations from exact SHAP values

`api/narrative.py` builds the explanation text from a fixed template filled with the exact numbers returned by `shap.TreeExplainer`. A check function verifies that every number in the text comes from the SHAP output, so the text cannot contain an invented figure. SHAP values are in the model's own output units (log-odds for gradient boosting).

## PostgreSQL (optional)

By default Django uses SQLite and needs no configuration. To use PostgreSQL for Django's tables and for the audit log of predictions (`api_predictionlog`) and drift checks (`api_driftchecklog`):

```powershell
$env:USE_POSTGRES = "1"
$env:POSTGRES_PASSWORD = "your_password"
# optional: POSTGRES_DB (default mlops_churn), POSTGRES_USER (postgres), POSTGRES_HOST (localhost), POSTGRES_PORT (5432)
pipenv run python manage.py migrate
```

Create an empty database named `mlops_churn` first. The password is read from the environment and is never stored in the code. MLflow keeps its own data in `mlflow.db` and `mlartifacts/`, separate from Django's database.

## Example results

| Model | Validation ROC-AUC |
|---|---|
| Logistic regression | 0.816 (baseline, not served) |
| Gradient boosting | 0.796 (`@champion`) |
| Random forest | 0.792 (`@challenger`) |

Logistic regression scores slightly higher but cannot be explained with `shap.TreeExplainer`, so only tree models compete for `@champion`.

Drift workflow on a shifted test batch: PSI flagged `monthly_charges` (1.807) and `support_tickets` (6.642) as significant while the other features stayed stable. The retrained candidate scored F1 0.7154 and AUC 0.7935 on the holdout against F1 0.7179 and AUC 0.7963 for the champion, so the gate **did not promote** it.

Example SHAP waterfall: `docs/waterfall_api.png`

## Limitations

- The holdout is a single fixed split, so repeated gate runs reuse the same data. A production system would refresh it.
- The drift endpoint reports PSI; retraining is started by running `training/workflow.py`, not by the endpoint itself.
- The dataset is synthetic.
- Models are saved with cloudpickle, which should only be loaded from trusted sources (here, your own MLflow server).

## Academic project

DS1, Subject #35: End-to-End MLOps Lifecycle with MLflow, Model Registry and SHAP.