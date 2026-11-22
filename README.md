# MLOps Project #35 — End-to-End Customer Churn Prediction

An end-to-end MLOps project implementing a complete machine learning lifecycle for B2B customer churn prediction.

## Project Overview

This project demonstrates an end-to-end MLOps workflow using:

- **Python** — Machine Learning development
- **Scikit-learn** — Model training and evaluation
- **MLflow** — Experiment tracking and Model Registry
- **Django REST Framework** — Prediction API
- **SHAP** — Model explainability
- **Pandas** — Data processing
- **Data Drift Monitoring** — PSI-based monitoring
- **Pipenv** — Dependency and virtual environment management

## Project Structure

- `api/` — Prediction and explanation API
- `config/` — Django configuration
- `data/` — Datasets
- `monitoring/` — Data drift monitoring
- `training/` — Model training, registration and promotion gate
- `manage.py` — Django application entry point
- `Pipfile` / `Pipfile.lock` — Dependencies (replace `requirements.txt`)

## Installation (Pipenv)

Clone the repository:

```bash
git clone https://github.com/eyabenabdelhedi/MLOps_Project35.git
cd MLOps_Project35
```

Install Pipenv (once):

```bash
pip install pipenv
```

On Windows, if `pip` or `pipenv` is not recognized, use `py -m pip install --user pipenv` and write `py -m pipenv` instead of `pipenv` in every command below.

Create the environment and install all dependencies from `Pipfile.lock`:

```bash
pipenv install --dev
```

Run any command inside the environment with `pipenv run <command>`, or open a shell with `pipenv shell`.

## Running the Project

The MLflow tracking server and the Django server must run in **separate terminals**.

| Terminal | Command | What it does |
|---|---|---|
| 1 | `pipenv run mlflow-server` | Starts MLflow at http://127.0.0.1:5000 |
| 2 | `pipenv run train` | Trains the three candidate models |
| 2 | `pipenv run register` | Registers the models and sets the aliases |
| 2 | `pipenv run migrate` then `pipenv run serve` | Starts the Django API at http://127.0.0.1:8000 |
| 2 | `pipenv run drift` | PSI drift report on a stable and a drifted batch |
| 2 | `pipenv run gate` | Champion/challenger promotion gate |
| 2 | `pipenv run test` | Automated tests |

Django API: `http://127.0.0.1:8000/api/predict/` and `http://127.0.0.1:8000/api/explain/`

MLflow UI: `http://127.0.0.1:5000/`

## Contributors

Add your team members here.

## Academic Project

DS1 — Subject #35  
End-to-End MLOps Lifecycle with MLflow, Model Registry and SHAP.