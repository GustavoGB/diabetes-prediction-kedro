"""Inference node behaviour: the adapter, the threshold and the feature contract."""

import numpy as np
import pandas as pd
import pytest

from diabetes.pipelines.inference import nodes


class StubEstimator:
    """Returns a probability read straight from a column, so threshold
    behaviour can be asserted exactly."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = features["SCORE"].to_numpy(dtype=float)
        return np.column_stack([1.0 - positive, positive])


@pytest.fixture
def model():
    return {
        "name": "StubEstimator",
        "estimator": StubEstimator(),
        "target_column": "OUTCOME",
        "feature_columns": ["SCORE"],
    }


@pytest.fixture
def scored_frame():
    return pd.DataFrame({"SCORE": [0.10, 0.50, 0.90], "OUTCOME": [0, 1, 1]})


def test_to_dataframe_passes_a_frame_through():
    frame = pd.DataFrame({"A": [1]})
    assert nodes.to_dataframe(frame) is frame


def test_to_dataframe_wraps_an_api_payload():
    """The adapter is the seam that lets one pipeline serve a CSV and JSON."""
    payload = [{"Glucose": 148, "Age": 50}, {"Glucose": 85, "Age": 31}]
    frame = nodes.to_dataframe(payload)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["Glucose", "Age"]
    assert len(frame) == 2


def test_predict_returns_one_record_per_row(model, scored_frame):
    predictions = nodes.predict(model, scored_frame, {"threshold": 0.5})
    assert len(predictions) == len(scored_frame)
    assert [p["row"] for p in predictions] == [0, 1, 2]
    assert [p["probability"] for p in predictions] == [0.1, 0.5, 0.9]


def test_predict_honours_the_threshold(model, scored_frame):
    at_half = nodes.predict(model, scored_frame, {"threshold": 0.5})
    assert [p["prediction"] for p in at_half] == [0, 1, 1]

    strict = nodes.predict(model, scored_frame, {"threshold": 0.95})
    assert [p["prediction"] for p in strict] == [0, 0, 0]

    permissive = nodes.predict(model, scored_frame, {"threshold": 0.05})
    assert [p["prediction"] for p in permissive] == [1, 1, 1]


def test_predict_uses_only_the_artifact_feature_columns(model, scored_frame):
    """The inference CSV still carries OUTCOME; it must never reach the model."""
    noisy = scored_frame.assign(UNEXPECTED="junk")
    assert nodes.predict(model, noisy, {"threshold": 0.5}) == nodes.predict(
        model, scored_frame, {"threshold": 0.5}
    )


def test_predict_preserves_the_source_row_index(model):
    """Row ids must map back to the caller's rows, not to a reset range."""
    frame = pd.DataFrame({"SCORE": [0.9, 0.1]}, index=[7, 9])
    predictions = nodes.predict(model, frame, {"threshold": 0.5})
    assert [p["row"] for p in predictions] == [7, 9]
