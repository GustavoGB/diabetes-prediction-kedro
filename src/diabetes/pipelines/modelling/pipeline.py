"""Baseline vs tuned model, evaluated on the held-out split, best one promoted."""

from kedro.pipeline import Node, Pipeline

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=nodes.train_model,
                inputs=["master_table", "params:baseline"],
                outputs="baseline_model",
                name="train_baseline_model",
            ),
            Node(
                func=nodes.evaluate_model,
                inputs=["baseline_model", "master_table"],
                outputs="baseline_metrics",
                name="evaluate_baseline_model",
            ),
            Node(
                func=nodes.tune_model,
                inputs=["master_table", "params:tuning"],
                outputs="tuned_model",
                name="tune_model",
            ),
            Node(
                func=nodes.evaluate_model,
                inputs=["tuned_model", "master_table"],
                outputs="tuned_metrics",
                name="evaluate_tuned_model",
            ),
            Node(
                func=nodes.select_best_model,
                inputs=[
                    "baseline_model",
                    "tuned_model",
                    "baseline_metrics",
                    "tuned_metrics",
                    "params:selection",
                ],
                outputs="production_model",
                name="select_best_model",
            ),
        ]
    )
