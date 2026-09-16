"""Inference-only nodes. Every transform is imported from data_engineering."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def to_dataframe(data: Any) -> pd.DataFrame:
    """Adapter so the same pipeline accepts a catalog CSV or an API payload."""
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def predict(
    model: dict[str, Any], data: pd.DataFrame, params: dict[str, Any]
) -> list[dict[str, Any]]:
    features = data[model["feature_columns"]]
    probabilities = model["estimator"].predict_proba(features)[:, 1]
    threshold = params["threshold"]

    logger.info("scored %d rows with %s", len(features), model["name"])
    return [
        {
            "row": int(index),
            "probability": round(float(probability), 6),
            "prediction": int(probability >= threshold),
        }
        for index, probability in zip(data.index, probabilities, strict=True)
    ]
