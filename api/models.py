from django.db import models


class PredictionLog(models.Model):
    """One row per scoring request (/api/predict/ or /api/explain/)."""
    created_at = models.DateTimeField(auto_now_add=True)
    endpoint = models.CharField(max_length=20)
    model_version = models.CharField(max_length=20)
    churn_probability = models.FloatField()
    input_data = models.JSONField()
    top_drivers = models.JSONField(null=True, blank=True)   # exact SHAP values
    narrative = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]


class DriftCheckLog(models.Model):
    """One row per production batch checked at /api/drift/check/."""
    created_at = models.DateTimeField(auto_now_add=True)
    n_rows = models.IntegerField()
    max_psi = models.FloatField()
    status = models.CharField(max_length=10)      # ok / warning / alert
    features = models.JSONField()                 # PSI and level per feature

    class Meta:
        ordering = ["-created_at"]