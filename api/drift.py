"""
POST /api/drift/check/  ->  PSI drift report for a batch of production customers.

Send:   {"customers": [ {tenure_months: .., ... company_size: ..}, ... ]}   (at least 50 customers)
Get:    PSI per feature, an overall status, and whether retraining is recommended.
"""

from pathlib import Path

import pandas as pd
from django.conf import settings
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView

from monitoring.psi import psi_report

REFERENCE_PATH = Path(settings.BASE_DIR) / "data" / "reference.csv"
MIN_ROWS = 50      # PSI on a handful of customers is just noise


class DriftBatchSerializer(serializers.Serializer):
    customers = serializers.ListField(
        child=serializers.DictField(), min_length=MIN_ROWS, max_length=20000
    )


def check_batch(customers, reference):
    """Pure function (easy to test): compare a list of customer dicts with the reference table."""
    batch = pd.DataFrame(customers)

    missing = [c for c in reference.columns if c not in batch.columns]
    if missing:
        raise ValueError(f"Missing columns in the batch: {missing}")
    batch = batch[list(reference.columns)]

    # Numbers arrive as JSON: make sure numeric columns are numbers
    for column in reference.columns:
        if pd.api.types.is_numeric_dtype(reference[column]):
            batch[column] = pd.to_numeric(batch[column], errors="coerce")

    report = psi_report(reference, batch)
    alerts = int((report["status"] == "ALERT").sum())
    warnings = int((report["status"] == "WARNING").sum())
    overall = "ALERT" if alerts else "WARNING" if warnings else "STABLE"

    return {
        "rows": int(len(batch)),
        "overall_status": overall,
        "alert_count": alerts,
        "warning_count": warnings,
        "retrain_recommended": alerts > 0,
        "features": report.to_dict(orient="records"),
    }


class DriftCheckView(APIView):
    """POST /api/drift/check/"""

    def post(self, request):
        serializer = DriftBatchSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({"success": False, "errors": serializer.errors},
                            status=status.HTTP_400_BAD_REQUEST)
        try:
            reference = pd.read_csv(REFERENCE_PATH)
            result = check_batch(serializer.validated_data["customers"], reference)
        except FileNotFoundError:
            return Response({"success": False,
                             "error": "data/reference.csv not found. Run training/train.py first."},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except ValueError as error:
            return Response({"success": False, "error": str(error)},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response({"success": True, **result}, status=status.HTTP_200_OK)