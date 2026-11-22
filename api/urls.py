from django.urls import path

from .drift import DriftCheckView
from .views import CustomerExplanationView, CustomerPredictionView

urlpatterns = [
    path("predict/", CustomerPredictionView.as_view(), name="predict"),
    path("explain/", CustomerExplanationView.as_view(), name="explain"),
    path("drift/check/", DriftCheckView.as_view(), name="drift-check"),
]