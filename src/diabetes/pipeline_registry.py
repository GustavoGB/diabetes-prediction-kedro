"""Pipeline registry."""

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

TRAINING = ("data_engineering", "modelling")
ALL = ("data_engineering", "modelling", "inference")


def _combine(pipelines: dict[str, Pipeline], names: tuple[str, ...]) -> Pipeline:
    return sum((pipelines[name] for name in names), Pipeline([]))


def register_pipelines() -> dict[str, Pipeline]:
    pipelines = find_pipelines(raise_errors=True)
    # `kedro run --pipeline inference` imports only that module, so guard the lookups.
    if all(name in pipelines for name in ALL):
        pipelines["training"] = _combine(pipelines, TRAINING)
        pipelines["__default__"] = _combine(pipelines, ALL)
    return pipelines
