"""AI narrative with a verification gate.

The LLM only sees exact numbers from explain_customer(). Its text is accepted
only if every number and every feature name in it appears in those facts.
Anything else (no key, API error, unverified text) falls back to the template.
"""
import json
import logging
import os
import re

log = logging.getLogger(__name__)

NUMBER = re.compile(r"(?<![\w.])([-+]?)(\d+(?:\.\d+)?)")
FEATURE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")

SYSTEM = """You explain one customer's churn prediction to a business user.
Rules:
- Use ONLY the numbers in the JSON facts. Copy them exactly as written (keep the
  % sign and the +/- sign). Never round, convert, sum, compare or invent numbers.
- Do not state counts, ranks or percentages that are not in the facts.
- Mention the churn probability and the average probability.
- Name each driver by its exact feature name, with its value and SHAP number.
- Say which drivers raise and which lower the risk, using the 'direction' field.
- Mention the SHAP unit once.
- Do not speculate about causes or give advice. 3 to 5 sentences, plain text."""


def build_facts(result, top_n=6):
    drivers = result["contributions"][:top_n]
    return {
        "churn_probability": f"{result['churn_probability'] * 100:.1f}%",
        "average_probability": f"{result['base_probability'] * 100:.1f}%",
        "shap_unit": result["unit"],
        "drivers": [
            {"feature": c["feature"],
             "value": f"{c['value']:g}",
             "shap": f"{c['shap_value']:+.3f}",
             "direction": ("raises churn risk" if c["shap_value"] > 0
                           else "lowers churn risk")}
            for c in drivers
        ],
    }


def _allowed(facts):
    """Numbers (absolute and signed) and feature names the text may contain."""
    absolute, signed, features = set(), set(), set()

    def add(s, with_sign=False):
        for sign, num in NUMBER.findall(s):
            v = float(num)
            absolute.add(round(v, 6))
            if with_sign:
                signed.add(round(float(sign + num), 6))

    add(facts["churn_probability"])
    add(facts["average_probability"])
    for d in facts["drivers"]:
        add(d["value"])
        add(d["shap"], with_sign=True)
        features.add(d["feature"])
    return absolute, signed, features


def verify(text, facts):
    """Return (ok, problems). Checks numbers, feature names and key facts."""
    absolute, signed, features = _allowed(facts)
    problems = []

    for sign, num in NUMBER.findall(text):
        v = round(float(num), 6)
        if sign:
            if round(float(sign + num), 6) not in signed and v not in absolute:
                problems.append(f"number not in facts: {sign}{num}")
        elif v not in absolute:
            problems.append(f"number not in facts: {num}")

    for name in FEATURE.findall(text):
        if name not in features:
            problems.append(f"unknown feature: {name}")

    if facts["churn_probability"].rstrip("%") not in text:
        problems.append("churn probability is not stated")
    return (not problems), problems


def _llm_text(facts):
    import anthropic
    client = anthropic.Anthropic(timeout=20)
    msg = client.messages.create(
        model=os.environ.get("NARRATIVE_MODEL", "claude-sonnet-5-5"),
        max_tokens=400,
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(facts)}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def build_narrative(result, top_n=6):
    """Return {'text', 'source', 'issues'}. source is 'llm' or 'template'."""
    template = result["narrative"]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"text": template, "source": "template", "issues": ["no API key set"]}

    facts = build_facts(result, top_n)
    try:
        text = _llm_text(facts)
    except Exception as exc:
        log.warning("Narrative LLM call failed: %s", exc)
        return {"text": template, "source": "template", "issues": [f"LLM error: {exc}"]}

    ok, problems = verify(text, facts)
    if ok:
        return {"text": text, "source": "llm", "issues": []}
    log.warning("Narrative rejected by verifier: %s", problems)
    return {"text": template, "source": "template", "issues": problems}