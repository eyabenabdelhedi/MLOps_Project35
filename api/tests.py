from django.test import SimpleTestCase

from api.narrative import build_facts, verify

RESULT = {
    "churn_probability": 0.723,
    "base_probability": 0.30,
    "unit": "probability points",
    "contributions": [
        {"feature": "tenure_months", "value": 8.0, "shap_value": 0.21},
        {"feature": "support_tickets", "value": 5.0, "shap_value": -0.05},
    ],
}


class VerifyNarrativeTests(SimpleTestCase):
    def setUp(self):
        self.facts = build_facts(RESULT)

    def test_facts_are_formatted_exactly(self):
        self.assertEqual(self.facts["churn_probability"], "72.3%")
        self.assertEqual(self.facts["drivers"][0]["shap"], "+0.210")
        self.assertEqual(self.facts["drivers"][1]["direction"], "lowers churn risk")

    def test_accepts_text_that_uses_only_the_facts(self):
        text = ("The churn probability is 72.3%, against an average of 30.0%. "
                "tenure_months (8) raises the risk (+0.210). "
                "support_tickets (5) lowers it (-0.050).")
        ok, problems = verify(text, self.facts)
        self.assertTrue(ok, problems)

    def test_rejects_an_invented_number(self):
        text = "The churn probability is 72.3%, but another 99.9% of customers stay."
        ok, problems = verify(text, self.facts)
        self.assertFalse(ok)
        self.assertIn("number not in facts: 99.9", problems)

    def test_rejects_an_unknown_feature(self):
        text = "The churn probability is 72.3% and age_group matters."
        ok, problems = verify(text, self.facts)
        self.assertFalse(ok)
        self.assertIn("unknown feature: age_group", problems)

    def test_rejects_text_without_the_probability(self):
        ok, problems = verify("tenure_months raises the risk.", self.facts)
        self.assertFalse(ok)
        self.assertIn("churn probability is not stated", problems)