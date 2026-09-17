"""Dataset-level validation: the checks Pydantic structurally cannot make."""

from pathlib import Path

import pandas as pd
import pytest
from kedro.config import OmegaConfigLoader

from diabetes.pipelines.data_engineering.nodes import clean_data
from diabetes.validation import validate_data

SUITE = {
    "suite": "test_suite",
    "fail_on_error": True,
    "expectations": [
        {"type": "ExpectTableRowCountToBeBetween", "kwargs": {"min_value": 3}},
        {
            "type": "ExpectColumnValuesToBeBetween",
            "kwargs": {"column": "GLUCOSE", "min_value": 1, "max_value": 300},
        },
        {
            "type": "ExpectColumnValuesToNotBeNull",
            "kwargs": {"column": "INSULIN", "mostly": 0.4},
        },
        {
            "type": "ExpectColumnMeanToBeBetween",
            "kwargs": {"column": "OUTCOME", "min_value": 0.2, "max_value": 0.5},
        },
    ],
}


@pytest.fixture
def healthy():
    return pd.DataFrame(
        {
            "GLUCOSE": [148.0, 85.0, 183.0, 137.0, 116.0, 78.0],
            "INSULIN": [120.0, 94.0, None, 168.0, None, 88.0],
            "OUTCOME": [1, 0, 1, 0, 0, 0],
        }
    )


def test_a_healthy_batch_passes_and_is_returned_unchanged(healthy):
    data, report = validate_data(healthy, SUITE)
    assert report["success"] is True
    assert report["n_failed"] == 0
    assert report["n_expectations"] == 4
    assert report["n_rows"] == 6
    pd.testing.assert_frame_equal(data, healthy)


def test_out_of_range_values_fail(healthy):
    """A physiologically impossible reading stops the run."""
    broken = healthy.assign(GLUCOSE=[148.0, 85.0, 9999.0, 137.0, 116.0, 78.0])
    with pytest.raises(ValueError, match="test_suite"):
        validate_data(broken, SUITE)


def test_a_degraded_feed_fails_even_though_every_row_is_valid(healthy):
    """The point of dataset-level validation: each row is individually fine,
    the batch is not. No per-record model can express this."""
    degraded = healthy.assign(INSULIN=[120.0, None, None, None, None, None])
    with pytest.raises(ValueError, match="test_suite"):
        validate_data(degraded, SUITE)


def test_class_imbalance_drift_fails(healthy):
    """Every OUTCOME is a legal 0 or 1; the distribution is what broke."""
    drifted = healthy.assign(OUTCOME=[0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="test_suite"):
        validate_data(drifted, SUITE)


def test_truncated_batch_fails(healthy):
    with pytest.raises(ValueError, match="test_suite"):
        validate_data(healthy.head(2), SUITE)


def test_fail_on_error_false_reports_instead_of_raising(healthy):
    """Useful for the inference branch when you would rather score and alert."""
    lenient = {**SUITE, "fail_on_error": False}
    data, report = validate_data(healthy.head(2), lenient)

    assert report["success"] is False
    assert report["n_failed"] >= 1
    assert (
        report["failures"][0]["expectation"] == "expect_table_row_count_to_be_between"
    )
    pd.testing.assert_frame_equal(data, healthy.head(2))


def test_expectations_on_absent_columns_are_skipped(healthy):
    """One suite must guard both the labelled CSV and an unlabelled payload."""
    unlabelled = healthy.drop(columns=["OUTCOME"])
    _, report = validate_data(unlabelled, SUITE)

    assert report["success"] is True
    assert report["n_expectations"] == 3, "the OUTCOME expectation should be skipped"


def test_a_table_level_expectation_is_narrowed_not_failed(healthy):
    """The skip rule has to reach column_set, not just column.

    ExpectTableColumnsToMatchSet declares its columns under ``column_set``, so
    a rule that only inspects ``column`` leaves OUTCOME required — and every
    genuinely unlabelled inference batch fails on a label it is not supposed
    to carry. Narrowing costs nothing here: ``clean_data`` already raises on a
    missing *feature* column, before validation ever runs, and says so more
    clearly than an expectation report would.
    """
    suite = {
        **SUITE,
        "expectations": [
            {
                "type": "ExpectTableColumnsToMatchSet",
                "kwargs": {
                    "column_set": ["GLUCOSE", "INSULIN", "OUTCOME"],
                    "exact_match": False,
                },
            }
        ],
    }

    _, report = validate_data(healthy.drop(columns=["OUTCOME"]), suite)
    assert report["success"] is True, report["failures"]
    assert report["n_expectations"] == 1, "narrowed, not dropped"

    # Nothing left to check: the expectation is dropped rather than run empty.
    _, report = validate_data(healthy[["OUTCOME"]].drop(columns=["OUTCOME"]), suite)
    assert report["n_expectations"] == 0


def test_the_shipped_cleaned_suite_accepts_an_unlabelled_batch():
    """The real conf/base suite, not a hand-written one, on an unlabelled feed.

    This is the regression that matters: scoring a batch that carries no label
    is the entire point of the inference branch, and it used to abort the run.
    """
    params = OmegaConfigLoader(conf_source=str(Path("conf"))).get("parameters")
    suite = {**params["data_quality"]["cleaned"], "fail_on_error": True}

    raw = pd.read_csv(Path("data/01_raw/diabetes-dataset-inference.csv"))
    cleaned = clean_data(raw, params["columns"])
    assert "OUTCOME" in cleaned.columns, "fixture assumption: this CSV is labelled"

    _, report = validate_data(cleaned.drop(columns=["OUTCOME"]), suite)
    assert report["success"] is True, report["failures"]
