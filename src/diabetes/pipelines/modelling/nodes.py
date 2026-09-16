"""Training, evaluation and model selection.

The estimator is a ``class_path`` string in YAML, so swapping LogisticRegression
for RandomForest / HistGradientBoosting / LightGBM is a one-line config change
with no code edit.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

SPLIT = "SPLIT"


def _load_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _feature_columns(data: pd.DataFrame, target: str) -> list[str]:
    return [col for col in data.columns if col not in (target, SPLIT)]


def _rows(data: pd.DataFrame, splits: list[str]) -> pd.DataFrame:
    return data[data[SPLIT].isin(splits)]


def train_model(master_table: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Fit the baseline estimator and return a self-describing artifact.

    ``feature_columns`` travels with the estimator so inference never has to
    guess the column order.
    """
    target = params["target_column"]
    features = _feature_columns(master_table, target)
    train = _rows(master_table, params["train_splits"])

    estimator = _load_class(params["class_path"])(**params.get("init_args", {}))
    estimator.fit(train[features], train[target])

    return {
        "name": params["class_path"].rsplit(".", 1)[-1],
        "estimator": estimator,
        "target_column": target,
        "feature_columns": features,
        "eval_splits": params["eval_splits"],
    }


def tune_model(master_table: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Grid-search on the train split only, so test metrics stay honest."""
    target = params["target_column"]
    features = _feature_columns(master_table, target)
    train = _rows(master_table, params["train_splits"])

    search = GridSearchCV(
        estimator=_load_class(params["class_path"])(**params.get("init_args", {})),
        param_grid=params["param_grid"],
        cv=params["cv"],
        scoring=params["scoring"],
        n_jobs=-1,
    ).fit(train[features], train[target])

    logger.info(
        "best params: %s (cv %s=%.4f)",
        search.best_params_,
        params["scoring"],
        search.best_score_,
    )
    return {
        "name": f"{params['class_path'].rsplit('.', 1)[-1]} (tuned)",
        "estimator": search.best_estimator_,
        "target_column": target,
        "feature_columns": features,
        "eval_splits": params["eval_splits"],
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
    }


def evaluate_model(model: dict[str, Any], master_table: pd.DataFrame) -> dict[str, Any]:
    """Per-split metrics.

    Note the argument order ``(y_true, y_pred)``: the notebook had it reversed
    everywhere, which silently swapped its precision and recall columns.
    """
    estimator = model["estimator"]
    target, features = model["target_column"], model["feature_columns"]

    report: dict[str, Any] = {"model": model["name"]}
    if "best_params" in model:
        report["best_params"] = model["best_params"]
        report["cv_best_score"] = model["cv_best_score"]

    for split in model["eval_splits"]:
        subset = _rows(master_table, [split])
        y_true = subset[target]
        y_pred = estimator.predict(subset[features])
        y_score = estimator.predict_proba(subset[features])[:, 1]
        report[split] = {
            "n_samples": int(len(subset)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
        }
        logger.info("%s | %s -> %s", model["name"], split, report[split])
    return report


def select_best_model(
    baseline_model: dict[str, Any],
    tuned_model: dict[str, Any],
    baseline_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Promote whichever candidate scores higher on the held-out split."""
    metric, split = params["metric"], params["split"]
    candidates = [
        (baseline_metrics[split][metric], baseline_model),
        (tuned_metrics[split][metric], tuned_model),
    ]
    score, winner = max(candidates, key=lambda pair: pair[0])
    logger.info("promoting %s (%s %s=%.4f)", winner["name"], split, metric, score)
    return {**winner, "selection": {"metric": metric, "split": split, "score": score}}
