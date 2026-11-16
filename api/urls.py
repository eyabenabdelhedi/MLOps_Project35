from django.urls import path

from .views import CustomerExplanationView, CustomerPredictionView

urlpatterns = [
    path("predict/", CustomerPredictionView.as_view(), name="predict"),
    path("explain/", CustomerExplanationView.as_view(), name="explain"),
]