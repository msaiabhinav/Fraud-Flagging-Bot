# slm.py
# SLM-style reasoning layer:
# - Calls model to get probability/decision
# - Produces a proper report-style explanation
# - For LogisticRegression: uses coefficient-based feature contributions (top drivers up/down)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from .pipeline import PredictionResult, predict, explain_prediction
except ImportError:
    from pipeline import PredictionResult, predict, explain_prediction


@dataclass
class TriageExplanation:
    decision: str
    fraud_probability: float
    threshold: float
    risk_level: str
    key_signals: List[str]
    data_quality_flags: List[str]
    summary: str


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def _norm_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _risk_level(p: float, threshold: float) -> str:
    if p >= max(0.85, threshold + 0.25):
        return "VERY_HIGH"
    if p >= max(0.70, threshold + 0.15):
        return "HIGH"
    if p >= threshold:
        return "MEDIUM"
    if p >= max(0.30, threshold - 0.15):
        return "LOW"
    return "VERY_LOW"


def _detect_signals(user_input: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Lightweight "business rule" signals for human-readable checks.
    These are NOT the model's true explanation; the coefficient explanation below is.
    """
    signals: List[str] = []
    flags: List[str] = []

    total = _to_float(user_input.get("total_claim_amount"))
    if total is not None:
        if total >= 20000:
            signals.append(f"High total_claim_amount (${total:,.0f})")
        elif total >= 10000:
            signals.append(f"Moderate-high total_claim_amount (${total:,.0f})")
        elif total <= 0:
            flags.append("total_claim_amount is non-positive")

    police = _norm_str(user_input.get("police_report_available"))
    if police and police.lower() in ("no", "n", "false", "0"):
        signals.append("Police report not available")

    w = _to_int(user_input.get("witnesses"))
    if w is not None:
        if w < 0:
            flags.append("witnesses is negative")
        elif w == 0:
            signals.append("No witnesses reported")

    nveh = _to_int(user_input.get("number_of_vehicles_involved"))
    if nveh is not None and nveh >= 3:
        signals.append(f"Multiple vehicles involved ({nveh})")

    # These are engineered fields only if you pass them in;
    # pipeline creates them internally, but user_input here is raw.
    hb = _norm_str(user_input.get("incident_hour_bucket"))
    if hb and hb.lower() == "night":
        signals.append("Incident time bucket = night")

    return signals, flags


def _pretty_feature_name(raw: str) -> str:
    """
    Convert ColumnTransformer+OHE feature names into something readable.
    Example: "insured_sex_MALE" -> "insured_sex = MALE"
             "num__policy_annual_premium" -> "policy_annual_premium"
    """
    s = str(raw)

    # remove transformer prefixes if present
    s = s.replace("num__", "").replace("cat__", "")

    # common OHE formatting:
    # sometimes it becomes "column_value" or "column=value"
    if "=" in s:
        return s

    # try split last underscore as category
    if "_" in s:
        parts = s.split("_")
        # if it looks like col + category (heuristic)
        if len(parts) >= 2:
            col = "_".join(parts[:-1])
            val = parts[-1]
            return f"{col} = {val}"

    return s


def _format_contrib(items: List[Dict[str, Any]], title: str, max_items: int = 6) -> str:
    if not items:
        return f"{title}: none detected."
    lines = [f"{title}:"]
    for it in items[:max_items]:
        feat = _pretty_feature_name(it.get("feature", ""))
        c = float(it.get("contribution", 0.0))
        lines.append(f" - {feat} (impact {c:+.4f})")
    return "\n".join(lines)


def triage(
    user_input: Dict[str, Any],
    *,
    model=None,
    preprocessor=None,
    threshold: float | None = None,
) -> TriageExplanation:
    """
    Main function used by chat_cli/bot_cli.
    Produces a report-style explanation with:
      - score + threshold
      - top features pushing risk up/down (LogisticRegression)
      - a short plain-English narrative
    """
    # model prediction
    result: PredictionResult = predict(
        user_input=user_input,
        model=model,
        preprocessor=preprocessor,
        threshold=threshold,
    )

    # business-rule signals (optional)
    signals, flags = _detect_signals(user_input)

    level = _risk_level(float(result.proba), float(result.threshold))

    # coefficient-based explanation (true model drivers for LR)
    exp = explain_prediction(
        user_input=user_input,
        top_k=10,
        model=model,
        preprocessor=preprocessor,
        threshold=threshold,
    )

    # Build a proper report summary
    headline = (
        "Flagged for manual review because the predicted risk is above the learned threshold."
        if result.label == 1
        else "Not flagged because the predicted risk is below the learned threshold."
    )

    pos_text = _format_contrib(exp.get("top_positive", []), "Main factors pushing risk UP")
    neg_text = _format_contrib(exp.get("top_negative", []), "Main factors pushing risk DOWN")

    # narrative: use top 3 drivers if available
    top_up = exp.get("top_positive", [])[:3]
    if top_up:
        drivers = ", ".join(_pretty_feature_name(d.get("feature", "")) for d in top_up)
        narrative = f"The model’s score is mainly driven upward by: {drivers}."
    else:
        narrative = "No strong single driver dominated the score; the risk is a combination of factors."

    # keep business signals as an extra section (optional)
    signals_line = "Heuristic checks: " + ("; ".join(signals[:5]) if signals else "none.")
    flags_line = "Data quality checks: " + ("; ".join(flags[:5]) if flags else "no obvious issues detected.")

    summary = (
        f"{headline}\n\n"
        f"Score details:\n"
        f"- Probability: {float(exp.get('probability', result.proba)):.4f}\n"
        f"- Threshold: {float(exp.get('threshold', result.threshold)):.4f}\n"
        f"- Risk level: {level}\n\n"
        f"{narrative}\n\n"
        f"{pos_text}\n\n"
        f"{neg_text}\n\n"
        f"{signals_line}\n"
        f"{flags_line}"
    )

    return TriageExplanation(
        decision=str(result.decision),
        fraud_probability=float(result.proba),
        threshold=float(result.threshold),
        risk_level=level,
        key_signals=signals,
        data_quality_flags=flags,
        summary=summary,
    )


def triage_to_dict(
    user_input: Dict[str, Any],
    *,
    model=None,
    preprocessor=None,
    threshold: float | None = None,
) -> Dict[str, Any]:
    return asdict(triage(user_input, model=model, preprocessor=preprocessor, threshold=threshold))
