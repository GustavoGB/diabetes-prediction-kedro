import pandas as pd
import pytest

from diabetes.pipelines.modelling import nodes

MODEL_PARAMS = {
    "class_path": "sklearn.linear_model.LogisticRegression",
    "init_args": {"random_state": 46, "max_iter": 1000},
    "target_column": "OUTCOME",
    "train_splits": ["train"],
    "eval_splits": ["train", "test"],
}


@pytest.fixture
def master_table():
    rows = 80
    return pd.DataFrame(
        {
            "OUTCOME": [i % 2 for i in range(rows)],
            "FEATURE_A": [i / rows for i in range(rows)],
            "FEATURE_B": [(i % 2) * 2.0 + 0.1 * (i % 5) for i in range(rows)],
            # both classes must land in each split: i % 4 in {0, 1} -> test
            "SPLIT": ["test" if i % 4 < 2 else "train" for i in range(rows)],
        }
    )


def test_artifact_carries_its_feature_contract(master_table):
    """feature_columns travels with the estimator, so inference never guesses
    the column order — and SPLIT/target are never treated as features."""
    model = nodes.train_model(master_table, MODEL_PARAMS)
    assert model["feature_columns"] == ["FEATURE_A", "FEATURE_B"]
    assert "SPLIT" not in model["feature_columns"]
    assert "OUTCOME" not in model["feature_columns"]


def test_estimator_class_comes_from_config(master_table):
    from sklearn.tree import DecisionTreeClassifier

    swapped = {**MODEL_PARAMS, "class_path": "sklearn.tree.DecisionTreeClassifier",
               "init_args": {"random_state": 46}}
    assert isinstance(nodes.train_model(master_table, swapped)["estimator"],
                      DecisionTreeClassifier)


def test_evaluate_reports_every_split(master_table):
    metrics = nodes.evaluate_model(nodes.train_model(master_table, MODEL_PARAMS),
                                   master_table)
    assert set(metrics) >= {"model", "train", "test"}
    assert metrics["train"]["n_samples"] + metrics["test"]["n_samples"] == len(master_table)
    for split in ("train", "test"):
        assert 0.0 <= metrics[split]["roc_auc"] <= 1.0


def test_selection_promotes_the_better_holdout_score(master_table):
    baseline = nodes.train_model(master_table, MODEL_PARAMS)
    tuned = {**baseline, "name": "challenger"}
    winner = nodes.select_best_model(
        baseline, tuned,
        {"test": {"roc_auc": 0.60}}, {"test": {"roc_auc": 0.90}},
        {"metric": "roc_auc", "split": "test"},
    )
    assert winner["name"] == "challenger"
    assert winner["selection"]["score"] == 0.90
