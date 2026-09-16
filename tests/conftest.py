import numpy as np
import pandas as pd
import pytest

PARAMS_COLUMNS = {
    "target": "OUTCOME",
    "raw_numerical": [
        "PREGNANCIES",
        "GLUCOSE",
        "BLOODPRESSURE",
        "SKINTHICKNESS",
        "INSULIN",
        "BMI",
        "DIABETESPEDIGREEFUNCTION",
        "AGE",
    ],
    "zero_as_missing": ["GLUCOSE", "BLOODPRESSURE", "SKINTHICKNESS", "INSULIN", "BMI"],
    "engineered_numerical": ["NEW_GLUCOSE_INSULIN", "NEW_GLUCOSE_PREGNANCIES"],
    "engineered_categorical": [
        "NEW_AGE_CAT",
        "NEW_BMI",
        "NEW_GLUCOSE",
        "NEW_AGE_BMI_NOM",
        "NEW_AGE_GLUCOSE_NOM",
    ],
}

FE_PARAMS = {
    "senior_age": 50,
    "bmi_bins": [0, 18.5, 24.9, 29.9, 100],
    "bmi_labels": ["Underweight", "Healthy", "Overweight", "Obese"],
    "glucose_bins": [0, 140, 200, 300],
    "glucose_labels": ["Normal", "Prediabetes", "Diabetes"],
    "glucose_band_edges": [69, 99, 125],
    "glucose_band_labels": ["low", "normal", "hidden", "high"],
}


@pytest.fixture
def columns():
    return PARAMS_COLUMNS


@pytest.fixture
def fe_params():
    return FE_PARAMS


@pytest.fixture
def raw_frame():
    """40 rows with the source schema, including sentinel zeros."""
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, n),
            "Glucose": rng.integers(80, 190, n),
            "BloodPressure": rng.integers(50, 100, n),
            "SkinThickness": rng.integers(10, 50, n),
            "Insulin": rng.integers(20, 300, n),
            "BMI": rng.uniform(19, 45, n).round(1),
            "DiabetesPedigreeFunction": rng.uniform(0.1, 1.5, n).round(3),
            "Age": rng.integers(21, 70, n),
            "Outcome": rng.integers(0, 2, n),
        }
    )
    frame.loc[0:4, "Insulin"] = 0  # the "not measured" sentinel
    frame.loc[5:6, "SkinThickness"] = 0
    return frame
