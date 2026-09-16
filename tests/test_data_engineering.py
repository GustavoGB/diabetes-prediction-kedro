import numpy as np
import pytest

from diabetes.pipelines.data_engineering import nodes

SPLIT_PARAMS = {"test_size": 0.3, "random_state": 17, "stratify": True}


def _prepared(raw_frame, columns):
    cleaned = nodes.clean_data(raw_frame, columns)
    return nodes.split_data(cleaned, columns, SPLIT_PARAMS)


def test_clean_data_maps_sentinel_zeros_to_nan(raw_frame, columns):
    cleaned = nodes.clean_data(raw_frame, columns)
    assert cleaned["INSULIN"].isna().sum() == 5
    assert cleaned["SKINTHICKNESS"].isna().sum() == 2
    # Pregnancies is not a sentinel column: its zeros are real.
    assert cleaned["PREGNANCIES"].isna().sum() == 0


def test_clean_data_tolerates_a_missing_target(raw_frame, columns):
    """The same function must serve an unlabelled inference payload."""
    unlabelled = raw_frame.drop(columns=["Outcome"])
    cleaned = nodes.clean_data(unlabelled, columns)
    assert "OUTCOME" not in cleaned.columns
    assert list(cleaned.columns) == columns["raw_numerical"]


def test_clean_data_rejects_a_missing_feature(raw_frame, columns):
    with pytest.raises(ValueError, match="missing required feature columns"):
        nodes.clean_data(raw_frame.drop(columns=["Glucose"]), columns)


def test_fit_nodes_only_see_the_train_split(raw_frame, columns):
    """Leakage guard: an absurd test-only value must not move the fitted scaler."""
    split = _prepared(raw_frame, columns)
    poisoned = split.copy()
    poisoned.loc[poisoned["SPLIT"] == "test", "AGE"] = 10_000

    featured = nodes.engineer_features(
        nodes.apply_imputer(
            split, nodes.fit_imputer(split, columns, {"n_neighbors": 3})
        ),
        {
            "senior_age": 50,
            "bmi_bins": [0, 18.5, 24.9, 29.9, 100],
            "bmi_labels": ["Underweight", "Healthy", "Overweight", "Obese"],
            "glucose_bins": [0, 140, 200, 300],
            "glucose_labels": ["Normal", "Prediabetes", "Diabetes"],
            "glucose_band_edges": [69, 99, 125],
            "glucose_band_labels": ["low", "normal", "hidden", "high"],
        },
    )
    poisoned_featured = featured.copy()
    poisoned_featured.loc[poisoned_featured["SPLIT"] == "test", "AGE"] = 10_000

    clean_scaler = nodes.fit_scaler(featured, columns)["scaler"]
    poisoned_scaler = nodes.fit_scaler(poisoned_featured, columns)["scaler"]
    np.testing.assert_allclose(clean_scaler.center_, poisoned_scaler.center_)


def test_fit_nodes_refuse_unsplit_data(raw_frame, columns):
    cleaned = nodes.clean_data(raw_frame, columns)
    with pytest.raises(ValueError, match="SPLIT column missing"):
        nodes.fit_imputer(cleaned, columns, {"n_neighbors": 3})


def test_encoder_schema_is_stable_for_a_single_row(raw_frame, columns, fe_params):
    """The notebook used pd.get_dummies, so a smaller batch silently produced a
    narrower matrix. The fitted encoder must always emit the same columns."""
    split = _prepared(raw_frame, columns)
    featured = nodes.engineer_features(
        nodes.apply_imputer(
            split, nodes.fit_imputer(split, columns, {"n_neighbors": 3})
        ),
        fe_params,
    )
    encoder = nodes.fit_encoder(featured, columns)

    full = nodes.apply_encoder(featured, encoder)
    one_row = nodes.apply_encoder(featured.head(1), encoder)
    assert list(full.columns) == list(one_row.columns)


def test_encoder_ignores_unseen_categories(raw_frame, columns, fe_params):
    split = _prepared(raw_frame, columns)
    featured = nodes.engineer_features(
        nodes.apply_imputer(
            split, nodes.fit_imputer(split, columns, {"n_neighbors": 3})
        ),
        fe_params,
    )
    encoder = nodes.fit_encoder(featured, columns)
    unseen = featured.head(1).copy()
    unseen["NEW_AGE_CAT"] = "martian"

    encoded = nodes.apply_encoder(unseen, encoder)
    assert list(encoded.columns) == list(nodes.apply_encoder(featured, encoder).columns)
    assert encoded.filter(like="NEW_AGE_CAT_").to_numpy().sum() == 0


def test_outlier_bounds_clip_both_tails(raw_frame, columns):
    split = _prepared(raw_frame, columns)
    imputed = nodes.apply_imputer(
        split, nodes.fit_imputer(split, columns, {"n_neighbors": 3})
    )
    bounds = nodes.fit_outlier_bounds(
        imputed, columns, {"q_low": 0.05, "q_high": 0.95, "factor": 1.5}
    )
    extreme = imputed.head(1).copy()
    extreme.loc[extreme.index[0], "AGE"] = 10_000

    capped = nodes.apply_outlier_bounds(extreme, bounds)
    assert capped["AGE"].iloc[0] == pytest.approx(bounds["AGE"][1])


def test_split_is_stratified(raw_frame, columns):
    split = _prepared(raw_frame, columns)
    rates = split.groupby("SPLIT")["OUTCOME"].mean()
    assert abs(rates["train"] - rates["test"]) < 0.15
    assert set(split["SPLIT"]) == {"train", "test"}
