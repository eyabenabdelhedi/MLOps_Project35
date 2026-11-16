import logging
import threading

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .explain import explain_customer


logger = logging.getLogger(__name__)

MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_NAME = "CustomerChurn"
MODEL_ALIAS = "champion"

# The seven features used when training the model
FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "monthly_usage_hours",
    "support_tickets",
    "payment_delays",
    "contract_type",
    "company_size",
]

NUMERIC_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "monthly_usage_hours",
    "support_tickets",
    "payment_delays",
]

# Same decision threshold as in training (train.py used >= 0.5)
THRESHOLD = 0.5


# Validate the customer information received by the API
class CustomerInputSerializer(serializers.Serializer):

    tenure_months = serializers.IntegerField(min_value=0)
    monthly_charges = serializers.FloatField(min_value=0)
    monthly_usage_hours = serializers.FloatField(min_value=0)
    support_tickets = serializers.IntegerField(min_value=0)
    payment_delays = serializers.IntegerField(min_value=0)

    contract_type = serializers.ChoiceField(
        choices=["monthly", "annual", "biennial"]
    )
    company_size = serializers.ChoiceField(
        choices=["small", "medium", "large"]
    )


# What we currently hold in memory: the model AND its version number
_model = None
_loaded_version = None
_lock = threading.Lock()   # prevents two requests from reloading at the same time


def get_champion_model():
    """
    Return (model, version) for the model the 'champion' alias points to.

    Each call asks MLflow which version the alias points to right now.
    If it changed, the new version is downloaded and replaces the old one.
    """

    global _model, _loaded_version

    mlflow.set_tracking_uri(MLFLOW_URI)

    try:
        client = MlflowClient()
        current_version = client.get_model_version_by_alias(
            MODEL_NAME, MODEL_ALIAS
        ).version
    except Exception:
        # MLflow is unreachable: keep serving the model we already have
        if _model is not None:
            logger.warning("Could not check the alias; using the cached model.")
            return _model, _loaded_version
        raise

    with _lock:
        if _model is None or current_version != _loaded_version:
            logger.info(
                "Loading %s@%s (version %s)", MODEL_NAME, MODEL_ALIAS, current_version
            )
            _model = mlflow.sklearn.load_model(
                f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            )
            _loaded_version = current_version

    return _model, _loaded_version


def build_customer_frame(validated_data):
    """Turn the validated JSON into the one-row table the model expects."""
    customer_df = pd.DataFrame([validated_data], columns=FEATURE_COLUMNS)
    customer_df[NUMERIC_COLUMNS] = customer_df[NUMERIC_COLUMNS].astype("float64")
    return customer_df


def error_response(message="The request could not be completed. Check the Django and MLflow server logs."):
    return Response(
        {"success": False, "error": message},
        status=status.HTTP_503_SERVICE_UNAVAILABLE
    )


class CustomerPredictionView(APIView):
    """POST /api/predict/  ->  churn probability."""

    def post(self, request):

        serializer = CustomerInputSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"success": False, "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            customer_df = build_customer_frame(serializer.validated_data)

            model, version = get_champion_model()

            churn_probability = float(model.predict_proba(customer_df)[0][1])
            prediction = int(churn_probability >= THRESHOLD)

            return Response(
                {
                    "success": True,
                    "model_name": MODEL_NAME,
                    "model_alias": MODEL_ALIAS,
                    "model_version": version,
                    "churn_probability": round(churn_probability, 4),
                    "prediction": prediction,
                    "label": "likely to leave" if prediction == 1 else "likely to stay"
                },
                status=status.HTTP_200_OK
            )

        except Exception:
            logger.exception("Customer prediction failed.")
            return error_response("The prediction could not be completed. Check the Django and MLflow server logs.")


class CustomerExplanationView(APIView):
    """POST /api/explain/  ->  SHAP contributions, waterfall plot and narrative."""

    def post(self, request):

        serializer = CustomerInputSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"success": False, "errors": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            customer_df = build_customer_frame(serializer.validated_data)

            model, version = get_champion_model()

            explanation = explain_customer(model, customer_df, version=version)
            prediction = int(explanation["churn_probability"] >= THRESHOLD)

            return Response(
                {
                    "success": True,
                    "model_name": MODEL_NAME,
                    "model_alias": MODEL_ALIAS,
                    "model_version": version,
                    "prediction": prediction,
                    **explanation,
                },
                status=status.HTTP_200_OK
            )

        except Exception:
            logger.exception("Customer explanation failed.")
            return error_response("The explanation could not be completed. Check the Django and MLflow server logs.")