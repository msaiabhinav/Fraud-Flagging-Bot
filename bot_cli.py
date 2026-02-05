# bot_cli.py
# CLI runner for Fraud Triage Bot

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

try:
    from .pipeline import load_dataset, sample_input_from_row, train_and_save
    from .slm import triage_to_dict
except ImportError:
    from pipeline import load_dataset, sample_input_from_row, train_and_save
    from slm import triage_to_dict


def _parse_kv_pairs(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"Invalid kv pair: '{p}'. Expected key=value.")
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()

        if v.lower() in ("none", "null", ""):
            out[k] = None
            continue

        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            pass

        out[k] = v
    return out


def _pretty_print(result: Dict[str, Any]) -> None:
    print("\n==============================")
    print(" FRAUD TRIAGE BOT (CLI)")
    print("==============================")
    print(f"Decision      : {result.get('decision')}")
    print(f"Risk Level    : {result.get('risk_level')}")
    print(f"Probability   : {result.get('fraud_probability'):.4f}")
    print(f"Threshold     : {result.get('threshold'):.4f}")

    print("\nKey Signals:")
    for s in (result.get("key_signals") or []):
        print(f" - {s}")
    if not (result.get("key_signals") or []):
        print(" - (none)")

    print("\nData Checks:")
    for f in (result.get("data_quality_flags") or []):
        print(f" - {f}")
    if not (result.get("data_quality_flags") or []):
        print(" - (none)")

    print("\nSummary:")
    print(" " + (result.get("summary") or ""))
    print("==============================\n")


def main():
    parser = argparse.ArgumentParser(description="Fraud Triage Bot CLI")
    parser.add_argument("--train", action="store_true", help="Train LR model + save artifacts")
    parser.add_argument("--from-row", type=int, default=None, help="Use a row index from insurance_claims.csv")
    parser.add_argument("--target-col", type=str, default="fraud_reported", help="Target column name in CSV")
    parser.add_argument("--json", type=str, default=None, help="JSON dict string of input fields")
    parser.add_argument("--kv", nargs="*", default=None, help="key=value pairs of input fields")
    parser.add_argument("--print-json", action="store_true", help="Print full JSON output too")
    args = parser.parse_args()

    if args.train:
        metrics = train_and_save(target_col=args.target_col)
        print("\nTraining complete. Saved artifacts to fraud_triage_bot/artifacts/")
        print(json.dumps(metrics, indent=2))
        return

    user_input: Optional[Dict[str, Any]] = None

    if args.from_row is not None:
        df = load_dataset()
        user_input = sample_input_from_row(df, row_idx=args.from_row, target_col=args.target_col)

    elif args.json is not None:
        try:
            user_input = json.loads(args.json)
            if not isinstance(user_input, dict):
                raise ValueError("JSON must decode into a dict/object.")
        except Exception as e:
            print(f"Failed to parse --json: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.kv is not None and len(args.kv) > 0:
        try:
            user_input = _parse_kv_pairs(args.kv)
        except Exception as e:
            print(f"Failed to parse --kv: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Provide one of: --train, --from-row, --json, or --kv key=value ...", file=sys.stderr)
        sys.exit(1)

    result = triage_to_dict(user_input)
    _pretty_print(result)

    if args.print_json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
