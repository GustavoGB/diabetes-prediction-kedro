"""Dataset-level validation with Great Expectations.

Pydantic guards the *request* boundary: one record, its fields and their types.
This guards the *dataset* boundary: how a whole batch is distributed — row
counts, null rates, value ranges, class balance. Those are properties no
single-row model can express, and they are where data pipelines actually fail.

Suites are declared in ``conf/base/parameters.yml`` as ``{type, kwargs}`` pairs
and instantiated by name, mirroring the ``class_path`` indirection the modelling
pipeline uses: a new check is a config change, not a code change.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.data_context.types.base import ProgressBarsConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _context():
    """One ephemeral GX context per process; nothing is written to disk."""
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = ProgressBarsConfig(
        globally=False, metric_calculations=False
    )
    return context


@lru_cache(maxsize=1)
def _batch_definition():
    return (
        _context()
        .data_sources.add_pandas("dataframes")
        .add_dataframe_asset(name="frame")
        .add_batch_definition_whole_dataframe("whole")
    )


def _build_suite(name: str, expectations: list[dict[str, Any]], columns: set[str]):
    """Instantiate the configured expectations, skipping absent columns.

    Skipping is deliberate: the same suite then guards the labelled modelling
    CSV and an unlabelled inference payload, exactly as ``clean_data`` does.
    An expectation suite must be registered with a context before it can run.
    """
    suite = _context().suites.add_or_update(gx.ExpectationSuite(name=name))
    for entry in expectations:
        kwargs = entry.get("kwargs", {})
        column = kwargs.get("column")
        if column is not None and column not in columns:
            logger.debug("skipping %s: column %s absent", entry["type"], column)
            continue
        suite.add_expectation(getattr(gx.expectations, entry["type"])(**kwargs))
    return suite


def validate_data(
    data: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a frame against its suite and pass it through unchanged.

    The frame is returned so downstream nodes depend on this one: a failed
    expectation stops the run instead of quietly poisoning the model.
    """
    suite = _build_suite(params["suite"], params["expectations"], set(data.columns))
    result = (
        _batch_definition()
        .get_batch(batch_parameters={"dataframe": data})
        .validate(suite)
    )

    failures = [
        {
            "expectation": outcome.expectation_config.type,
            "column": outcome.expectation_config.kwargs.get("column"),
            "unexpected_count": outcome.result.get("unexpected_count"),
            "unexpected_percent": outcome.result.get("unexpected_percent"),
            "observed_value": outcome.result.get("observed_value"),
        }
        for outcome in result.results
        if not outcome.success
    ]

    report = {
        "suite": params["suite"],
        "success": bool(result.success),
        "n_rows": int(len(data)),
        "n_expectations": len(result.results),
        "n_failed": len(failures),
        "failures": failures,
    }

    if failures:
        logger.warning("%s: %d expectation(s) failed", params["suite"], len(failures))
        for failure in failures:
            logger.warning("  %s", failure)
        if params.get("fail_on_error", True):
            raise ValueError(
                f"data quality suite '{params['suite']}' failed on {len(data)} rows: "
                f"{failures}"
            )
    else:
        logger.info(
            "%s: %d expectations passed on %d rows",
            params["suite"],
            len(result.results),
            len(data),
        )
    return data, report
