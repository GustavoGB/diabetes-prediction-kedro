"""Data engineering nodes, shared verbatim with the inference pipeline.

Every stateful step is a pair: ``fit_*`` produces an artifact the catalog
persists, ``apply_*`` consumes it and never refits. Training fits on the train
split only; inference imports the very same ``apply_*`` functions, so there is
exactly one implementation of clean / impute / engineer / encode / scale and
training-serving skew is structurally impossible.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

logger = logging.getLogger(__name__)

TRAIN = "train"
SPLIT = "SPLIT"


def _train_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Rows an artifact may be fitted on. Absent at inference time -> use all rows."""
    if SPLIT not in data.columns:
        raise ValueError(f"{SPLIT} column missing: fit nodes run on split data only")
    return data[data[SPLIT] == TRAIN]


def clean_data(data: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
    """Upper-case the schema, keep known columns, and map sentinel zeros to NaN.

    The target is dropped silently when absent, which is what lets this one
    function serve both the modelling CSV and an unlabelled inference payload.
    """
    df = data.copy()
    df.columns = [str(col).upper() for col in df.columns]

    keep = [columns["target"], *columns["raw_numerical"]]
    missing = [col for col in columns["raw_numerical"] if col not in df.columns]
    if missing:
        raise ValueError(f"input is missing required feature columns: {missing}")
    df = df[[col for col in keep if col in df.columns]]

    for col in columns["raw_numerical"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # 0 is this dataset's sentinel for "not measured" on these columns.
    for col in columns["zero_as_missing"]:
        df[col] = df[col].mask(df[col] == 0)

    logger.info("cleaned %d rows; NaN counts: %s", len(df), df.isna().sum().to_dict())
    return df


def split_data(
    data: pd.DataFrame, columns: dict[str, Any], params: dict[str, Any]
) -> pd.DataFrame:
    """Tag each row train/test. Stratified, unlike the notebook, because the
    target is imbalanced (~35% positive) and n is small."""
    df = data.copy()
    target = columns["target"]
    stratify = df[target] if params.get("stratify") else None
    _, test_index = train_test_split(
        df.index,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=stratify,
    )
    df[SPLIT] = TRAIN
    df.loc[test_index, SPLIT] = "test"
    logger.info("split sizes: %s", df[SPLIT].value_counts().to_dict())
    return df


def fit_imputer(
    data: pd.DataFrame, columns: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Scale -> KNN-impute -> inverse-scale, as in the notebook, but fit on the
    train split only (the notebook fit on all 652 rows: leakage)."""
    cols = columns["zero_as_missing"]
    train = _train_rows(data)
    scaler = RobustScaler().fit(train[cols])
    knn = KNNImputer(n_neighbors=params["n_neighbors"]).fit(
        scaler.transform(train[cols])
    )
    return {"columns": cols, "scaler": scaler, "knn": knn}


def apply_imputer(data: pd.DataFrame, imputer: dict[str, Any]) -> pd.DataFrame:
    df = data.copy()
    cols = imputer["columns"]
    scaled = imputer["scaler"].transform(df[cols])
    df[cols] = imputer["scaler"].inverse_transform(imputer["knn"].transform(scaled))
    return df


def fit_outlier_bounds(
    data: pd.DataFrame, columns: dict[str, Any], params: dict[str, Any]
) -> dict[str, list[float]]:
    """Tukey fences on the notebook's 5th/95th percentiles, train split only."""
    train = _train_rows(data)
    bounds: dict[str, list[float]] = {}
    for col in columns["raw_numerical"]:
        low = train[col].quantile(params["q_low"])
        high = train[col].quantile(params["q_high"])
        spread = (high - low) * params["factor"]
        bounds[col] = [float(low - spread), float(high + spread)]
    return bounds


def apply_outlier_bounds(
    data: pd.DataFrame, bounds: dict[str, list[float]]
) -> pd.DataFrame:
    df = data.copy()
    for col, (low, high) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(low, high)
    return df


def engineer_features(data: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """The notebook's derived columns. Stateless, so no fitted artifact."""
    df = data.copy()
    df["NEW_AGE_CAT"] = np.where(df["AGE"] >= params["senior_age"], "senior", "mature")

    bmi = pd.cut(
        df["BMI"],
        bins=[-np.inf, *params["bmi_bins"], np.inf],
        labels=params["bmi_labels"],
    )
    glucose = pd.cut(
        df["GLUCOSE"],
        bins=[-np.inf, *params["glucose_bins"], np.inf],
        labels=params["glucose_labels"],
    )
    df["NEW_BMI"] = bmi.astype(str)
    df["NEW_GLUCOSE"] = glucose.astype(str)

    # Glucose is integer-valued, so right-closed edges reproduce the notebook's
    # <70 / 70-99 / 100-125 / >125 bands exactly.
    band = pd.cut(
        df["GLUCOSE"],
        bins=[-np.inf, *params["glucose_band_edges"], np.inf],
        labels=params["glucose_band_labels"],
    ).astype(str)

    df["NEW_AGE_BMI_NOM"] = df["NEW_BMI"].str.lower() + df["NEW_AGE_CAT"]
    df["NEW_AGE_GLUCOSE_NOM"] = band + df["NEW_AGE_CAT"]
    df["NEW_GLUCOSE_INSULIN"] = df["GLUCOSE"] * df["INSULIN"]
    df["NEW_GLUCOSE_PREGNANCIES"] = df["GLUCOSE"] * df["PREGNANCIES"]
    return df


def fit_encoder(data: pd.DataFrame, columns: dict[str, Any]) -> dict[str, Any]:
    """One-hot encoder fitted on the train split.

    ``handle_unknown="ignore"`` is the whole point: the notebook used
    ``pd.get_dummies`` on the full frame, so scoring a smaller batch silently
    produced a different, shorter feature matrix.
    """
    cats = columns["engineered_categorical"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(_train_rows(data)[cats].astype(str))
    return {"columns": cats, "encoder": encoder}


def apply_encoder(data: pd.DataFrame, encoder: dict[str, Any]) -> pd.DataFrame:
    df = data.copy()
    cats, fitted = encoder["columns"], encoder["encoder"]
    dummies = pd.DataFrame(
        fitted.transform(df[cats].astype(str)),
        columns=fitted.get_feature_names_out(cats),
        index=df.index,
    )
    _warn_on_unseen_categories(df, cats, dummies, fitted)
    return pd.concat([df.drop(columns=cats), dummies], axis=1)


def _warn_on_unseen_categories(
    df: pd.DataFrame, cats: list[str], dummies: pd.DataFrame, fitted: Any
) -> None:
    """Log the rows whose category the encoder never saw while fitting.

    ``handle_unknown="ignore"`` keeps the feature schema stable, which is what
    inference needs, but it does so by encoding an unseen level as all zeros.
    That is indistinguishable from a legitimate row to every layer downstream:
    the model still scores it, confidently. Pydantic cannot catch it either —
    each field is individually in range; it is the *combination* that never
    occurred in training. So it is at least logged.
    """
    for column, levels in zip(cats, fitted.categories_):
        group = [name for name in dummies.columns if name.startswith(f"{column}_")]
        if not group:
            continue
        unseen = dummies.index[dummies[group].sum(axis=1) == 0]
        if len(unseen) == 0:
            continue
        observed = sorted(set(df.loc[unseen, column].astype(str)) - set(levels))
        logger.warning(
            "%s: %d row(s) carry a level the encoder never saw (%s); "
            "the whole %s group encodes as zeros",
            column,
            len(unseen),
            ", ".join(observed) or "unknown",
            column,
        )


def fit_scaler(data: pd.DataFrame, columns: dict[str, Any]) -> dict[str, Any]:
    cols = columns["raw_numerical"] + columns["engineered_numerical"]
    return {"columns": cols, "scaler": RobustScaler().fit(_train_rows(data)[cols])}


def apply_scaler(data: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    df = data.copy()
    cols = scaler["columns"]
    df[cols] = scaler["scaler"].transform(df[cols])
    return df
