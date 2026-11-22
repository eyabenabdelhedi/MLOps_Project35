from django.urls import path
from .ui import explain_page
from .views import DriftCheckView, ExplainView, PredictView

urlpatterns = [
    path("predict/", PredictView.as_view()),
    path("explain/", ExplainView.as_view()),
    path("drift/check/", DriftCheckView.as_view()),
    path("ui/", explain_page),
]