# chat_cli.py
# One-by-one intake for Fraud Triage Bot
# - Asks every required field sequentially (from insurance_claims.csv)
# - Shows accepted formats (YES/NO, number examples, date examples)
# - Handles common typos for YES/NO and SKIP via fuzzy matching (no big misspelling tables)
# - "skip / dont know / idk / ..." skips ONE field and moves on (including dates)
# - "go ahead / done / generate report" fills remaining defaults and generates report

from __future__ import annotations

from typing import Any, Dict, List
from difflib import SequenceMatcher

import pandas as pd
from dateutil import parser as dtparser

try:
    from .slm import triage_to_dict
    from .pipeline import DATASET_PATH
except ImportError:
    from slm import triage_to_dict
    from pipeline import DATASET_PATH


# --------------------------
# Required fields from Kaggle (minus dropped + target)
# --------------------------
DROP_COLS = {
    "policy_number",
    "auto_make",
    "auto_model",
    "incident_city",
    "incident_state",
    "insured_zip",
    "policy_state",
    "fraud_reported",
}


def load_required_fields(csv_path: str) -> List[str]:
    cols = list(pd.read_csv(csv_path, nrows=1).columns)
    return [c for c in cols if c not in DROP_COLS]


# --------------------------
# STOP words (finish & run report)
# --------------------------
GO_AHEAD_PHRASES = {
    "go ahead",
    "done",
    "generate report",
    "run report",
    "submit",
    "finished",
    "that's all",
    "that is all",
}


def user_wants_report(text: str) -> bool:
    return (text or "").strip().lower() in GO_AHEAD_PHRASES


# --------------------------
# Fuzzy intent (YES/NO/SKIP) without hardcoding misspellings
# --------------------------
def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def normalize_intent(text: str) -> str:
    """
    Returns one of:
      - "YES" / "NO"  (for yes/no answers)
      - "SKIP"        (for skip/don't know)
      - ""            (unknown intent)
    """
    t = (text or "").strip().lower()
    t = " ".join(t.split())
    if not t:
        return ""

    # exact quick hits
    if t in {"1", "yes", "y", "true"}:
        return "YES"
    if t in {"0", "no", "n", "false"}:
        return "NO"
    if t in {"skip", "idk", "unknown", "na", "n/a", "-", "dont know", "don't know", "not sure", "dunno"}:
        return "SKIP"

    # fuzzy compare to small intent vocab
    yes_words = ["yes", "yep", "yeah", "true"]
    no_words = ["no", "nope", "nah", "false"]
    skip_words = ["skip", "unknown", "dont know", "don't know", "idk", "not sure", "dunno"]

    best_yes = max(_sim(t, w) for w in yes_words)
    best_no = max(_sim(t, w) for w in no_words)
    best_skip = max(_sim(t, w) for w in skip_words)

    # thresholds: high enough to avoid false positives
    if best_skip >= 0.78:
        return "SKIP"
    if best_yes >= 0.78 and best_yes >= best_no:
        return "YES"
    if best_no >= 0.78:
        return "NO"
    return ""


def user_skipped(text: str) -> bool:
    return normalize_intent(text) == "SKIP"


# --------------------------
# Field type groups
# --------------------------
NUMERIC_FIELDS = {
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss",
    "incident_hour_of_the_day", "number_of_vehicles_involved",
    "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_year",
}

YESNO_FIELDS = {"police_report_available", "property_damage"}
DATE_FIELDS = {"incident_date", "policy_bind_date"}


# --------------------------
# Defaults (used ONLY on go-ahead OR skip)
# IMPORTANT: dates must be skippable -> store "Unknown" (not None)
# --------------------------
def default_for(field: str) -> Any:
    if field in NUMERIC_FIELDS:
        return 0
    if field in YESNO_FIELDS:
        return "Unknown"
    if field in DATE_FIELDS:
        return "Unknown"   # enables skipping dates without repeating
    return "Unknown"


def is_missing_value(v: Any) -> bool:
    # Treat "Unknown" as filled (so we move on)
    return v is None or (isinstance(v, str) and v.strip() == "")


def list_missing(state: Dict[str, Any], required: List[str]) -> List[str]:
    return [f for f in required if f not in state or is_missing_value(state[f])]


# --------------------------
# Parsing helpers
# --------------------------
def parse_value(field: str, text: str) -> Any:
    # If user indicates skip/unknown (even with typos) -> default and move on
    if user_skipped(text):
        return default_for(field)

    if field in YESNO_FIELDS:
        intent = normalize_intent(text)
        if intent in ("YES", "NO"):
            return intent
        return "Unknown"

    if field in DATE_FIELDS:
        try:
            return dtparser.parse(text, fuzzy=True).strftime("%Y-%m-%d")
        except Exception:
            return None

    if field in NUMERIC_FIELDS:
        try:
            return float(str(text).replace(",", ""))
        except Exception:
            return None

    return (text or "").strip() or "Unknown"


# --------------------------
# Better prompts (tell user what to type)
# --------------------------
FRIENDLY = {
    "police_report_available": "police report available",
    "property_damage": "property damage",
    "policy_bind_date": "policy bind date",
    "incident_date": "incident date",
    "total_claim_amount": "total claim amount",
    "injury_claim": "injury claim amount",
    "property_claim": "property claim amount",
    "vehicle_claim": "vehicle claim amount",
}

def question_for(field: str) -> str:
    name = FRIENDLY.get(field, field)

    if field in YESNO_FIELDS:
        return f"Please enter {name} (type: YES / NO). You can also type 1=yes, 0=no. Or 'skip'."

    if field in DATE_FIELDS:
        return f"Please enter {name} (e.g., 2009-12-12 or 12 Dec 2009). Or 'skip'."

    if field in NUMERIC_FIELDS:
        return f"Please enter {name} (numbers only, e.g., 18942). Or 'skip'."

    return f"Please enter {name}. Or 'skip'."


# --------------------------
# MAIN LOOP
# --------------------------
def main():
    required_fields = load_required_fields(DATASET_PATH)

    print("\n=== Fraud Triage Bot — Step-by-Step Intake ===")
    print("I will ask details one by one.")
    print("Type 'skip' / 'dont know' / 'idk' to skip a question (typos are okay).")
    print("Type 'go ahead' when you want the report.\n")

    state: Dict[str, Any] = {}

    while True:
        missing = list_missing(state, required_fields)

        # If all fields collected → auto report
        if not missing:
            result = triage_to_dict(state)
            print("\nBot (triage):")
            print(
                f"Decision: {result['decision']} | "
                f"P={result['fraud_probability']:.4f} | "
                f"Risk={result['risk_level']}"
            )
            print("Summary:", result["summary"])
            print("\nBot: Type 'new' to start again or 'exit' to quit.\n")

            cmd = input("You: ").strip().lower()
            if cmd in ("exit", "quit"):
                print("Bot: Bye.")
                break
            if cmd == "new":
                state = {}
                continue
            continue

        field = missing[0]
        user = input(f"Bot: {question_for(field)}\nYou: ").strip()

        if user.lower() in ("exit", "quit"):
            print("Bot: Bye.")
            break

        if user.lower() == "new":
            state = {}
            print("Bot: New claim started.\n")
            continue

        if user_wants_report(user):
            # Fill remaining with defaults and continue loop to output report
            for f in missing:
                state[f] = default_for(f)
            continue

        value = parse_value(field, user)

        # If parsing failed (not skip), re-ask same question
        if value is None and not user_skipped(user):
            print("Bot: I couldn’t understand that. Please try again or type 'skip'.\n")
            continue

        state[field] = value


if __name__ == "__main__":
    main()
