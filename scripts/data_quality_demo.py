"""Show what each validation layer can and cannot see.

Four realistic corruptions are pushed through both layers. In three of them
every individual row is valid — Pydantic is satisfied — and the dataset is
nonetheless unfit to train or score on.

    uv run python scripts/data_quality_demo.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd
import yaml
from pydantic import ValidationError

from diabetes.api import Patient
from diabetes.pipelines.data_engineering.nodes import clean_data
from diabetes.validation import validate_data

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

PROJECT = Path(__file__).resolve().parents[1]
PARAMS = yaml.safe_load((PROJECT / "conf" / "base" / "parameters.yml").read_text())
COLUMNS = PARAMS["columns"]
# Advisory mode: report every corruption instead of stopping at the first.
SUITE = {**PARAMS["data_quality"]["cleaned"], "fail_on_error": False}


def pydantic_verdict(frame: pd.DataFrame) -> str:
    invalid = 0
    for record in frame.drop(columns=["Outcome"], errors="ignore").to_dict("records"):
        try:
            Patient(**record)
        except ValidationError:
            invalid += 1
    return f"{len(frame) - invalid}/{len(frame)} rows valid"


def report(frame: pd.DataFrame, label: str) -> None:
    _, result = validate_data(clean_data(frame, COLUMNS), SUITE)
    print(f"\n{label}")
    print(f"  Pydantic : {pydantic_verdict(frame)}")
    print(
        f"  GX       : {'PASS' if result['success'] else 'FAIL'} "
        f"({result['n_failed']}/{result['n_expectations']} expectations failed)"
    )
    for failure in result["failures"]:
        observed = failure["observed_value"]
        detail = (
            observed
            if observed is not None
            else f"{failure['unexpected_percent']:.1f}% unexpected"
        )
        print(
            f"             - {failure['expectation']}({failure['column']}) -> {detail}"
        )


def main() -> None:
    raw = pd.read_csv(PROJECT / "data" / "01_raw" / "diabetes-dataset-inference.csv")

    report(raw, "0. The real inference batch (baseline)")
    report(raw.head(5), "1. Upstream job truncated the feed to 5 rows")

    degraded = raw.copy()
    degraded.loc[degraded.index[:105], "Insulin"] = 0
    report(degraded, "2. Insulin sensor stopped reporting (90% now unmeasured)")

    drifted = raw.copy()
    drifted["Outcome"] = 0
    drifted.loc[drifted.index[:3], "Outcome"] = 1
    report(drifted, "3. Label pipeline broke: positives collapse to 2.6%")

    units = raw.copy()
    units["BMI"] = units["BMI"] * 10
    report(units, "4. BMI arrives in the wrong unit (x10)")


if __name__ == "__main__":
    main()
