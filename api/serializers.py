
from rest_framework import serializers


class CustomerSerializer(serializers.Serializer):
    """
    Validates the information received about one customer.

    These fields must match the features used during model training.
    """

    # Numerical customer information
    tenure_months = serializers.IntegerField(min_value=0)

    monthly_charges = serializers.FloatField(min_value=0)

    monthly_usage_hours = serializers.FloatField(min_value=0)

    support_tickets = serializers.IntegerField(min_value=0)

    payment_delays = serializers.IntegerField(min_value=0)

    # Categorical customer information
    contract_type = serializers.CharField(
        max_length=50,
        trim_whitespace=True
    )

    company_size = serializers.CharField(
        max_length=50,
        trim_whitespace=True
    )
