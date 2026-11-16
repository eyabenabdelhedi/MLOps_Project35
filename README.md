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

## Project Structure

- `api/` — Prediction and explanation API
- `config/` — Django configuration
- `data/` — Datasets
- `monitoring/` — Data drift monitoring
- `training/` — Model training and registration
- `manage.py` — Django application entry point

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd MLOps_Project35
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The MLflow tracking server and Django server must run separately.

### 1. Start MLflow

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```

### 2. Train the models

```bash
python training/train.py
```

### 3. Register the models

Run the model registration script after configuring it for the newly generated MLflow runs:

```bash
python training/register.py
```

### 4. Start Django

```bash
python manage.py migrate
python manage.py runserver
```

Django API: `http://127.0.0.1:8000/`

MLflow UI: `http://127.0.0.1:5000/`

## Contributors

Add your team members here.

## Academic Project

DS1 — Subject #35  
End-to-End MLOps Lifecycle with MLflow, Model Registry and SHAP.
