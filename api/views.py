import logging
from pathlib import Path

import pandas as pd
from django.conf import settings
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from monitoring.psi_batch import psi_report

from . import services
from .models import DriftCheckLog, PredictionLog

log = logging.getLogger(__name__)


class CustomerSerializer(serializers.Serializer):
    tenure_months = serializers.FloatField(required=False, allow_null=True)
    monthly_charges = serializers.FloatField(required=False, allow_null=True)
    monthly_usage_hours = serializers.FloatField(required=False, allow_null=True)
    support_tickets = serializers.FloatField(required=False, allow_null=True)
    payment_delays = serializers.FloatField(required=False, allow_null=True)
    contract_type = serializers.CharField(required=False, allow_null=True)
    company_size = serializers.CharField(required=False, allow_null=True)


class _ScoringView(APIView):
    action = None      # set in subclasses
    endpoint = ""      # "predict" or "explain"

    def post(self, request):
        ser = CustomerSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = dict(ser.validated_data)
        try:
            result = self.action(data)
        except RuntimeError as exc:  # e.g. champion is not a tree model
            return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as exc:     # e.g. MLflow down or no @champion alias yet
            return Response({"error": f"Scoring unavailable: {exc}"},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        self._save(data, result)
        return Response(result)

    def _save(self, data, result):
        try:
            PredictionLog.objects.create(
                endpoint=self.endpoint,
                model_version=str(result["model_version"]),
                churn_probability=result["churn_probability"],
                input_data=data,
                top_drivers=result.get("top_drivers"),
                narrative=result.get("narrative", ""),
            )
        except Exception as exc:     # never break scoring because of the audit log
            log.warning("Could not save prediction log: %s", exc)


class PredictView(_ScoringView):
    action = staticmethod(services.predict)
    endpoint = "predict"


class ExplainView(_ScoringView):
    action = staticmethod(services.explain)
    endpoint = "explain"


class DriftCheckView(APIView):
    """POST {"records": [ {customer}, ... ]} -> PSI per feature + alert status."""

    def post(self, request):
        records = request.data.get("records")
        if not isinstance(records, list) or len(records) < 30:
            return Response({"error": "Send a 'records' list with at least 30 customers."},
                            status=status.HTTP_400_BAD_REQUEST)
        ref = pd.read_csv(Path(settings.BASE_DIR) / "data" / "reference.csv")
        report = psi_report(ref, pd.DataFrame(records), services.NUM, services.CAT)
        try:
            DriftCheckLog.objects.create(n_rows=report["n_rows"], max_psi=report["max_psi"],
                                         status=report["status"], features=report["features"])
        except Exception as exc:
            log.warning("Could not save drift log: %s", exc)
        return Response(report)