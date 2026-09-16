"""FastAPI service over the trained Kedro artifacts.

Artifacts are loaded from the Kedro catalog **once** at startup and scoring
calls the same node functions the batch pipeline uses, so an HTTP prediction and
a `kedro run --pipeline inference` prediction cannot diverge.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pydantic import BaseModel, Field

from diabetes.pipelines.data_engineering.nodes import (
    apply_encoder,
    apply_imputer,
    apply_outlier_bounds,
    apply_scaler,
    clean_data,
    engineer_features,
)
from diabetes.pipelines.inference.nodes import predict as predict_node

logger = logging.getLogger(__name__)

PROJECT_PATH = Path(__file__).resolve().parents[2]
ARTIFACTS = ("production_model", "imputer", "outlier_bounds", "encoder", "scaler")
EXPOSED_DATASETS = (
    "raw_modelling_data",
    "raw_inference_data",
    "master_table",
    "inference_predictions",
)

state: dict[str, Any] = {}


def _load_state() -> None:
    """Read artifacts and parameters from the Kedro catalog, once."""
    bootstrap_project(PROJECT_PATH)
    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        catalog = context.catalog
        state["params"] = context.params
        state["catalog"] = catalog
        for name in ARTIFACTS:
            state[name] = catalog.load(name)
    logger.info("loaded %s", ", ".join(ARTIFACTS))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_state()
    except Exception as exc:  # the API still starts, /ready reports 503
        state["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("artifacts unavailable — run `kedro run` first: %s", exc)
    yield
    state.clear()


app = FastAPI(
    title="Diabetes Prediction API",
    description="Serves the Kedro-trained diabetes incidence model.",
    version="0.1.0",
    lifespan=lifespan,
)


class Patient(BaseModel):
    """Raw measurements. 0 is the dataset's sentinel for 'not measured' on
    Glucose, BloodPressure, SkinThickness, Insulin and BMI — it is accepted and
    imputed, exactly as in the training pipeline."""

    model_config = {"extra": "forbid"}

    Pregnancies: int = Field(ge=0, le=20, examples=[6])
    Glucose: int = Field(ge=0, le=300, examples=[148])
    BloodPressure: int = Field(ge=0, le=200, examples=[72])
    SkinThickness: int = Field(ge=0, le=110, examples=[35])
    Insulin: int = Field(ge=0, le=1000, examples=[0])
    BMI: float = Field(ge=0, le=80, examples=[33.6])
    DiabetesPedigreeFunction: float = Field(gt=0, le=3, examples=[0.627])
    Age: int = Field(ge=21, le=120, examples=[50])


class Prediction(BaseModel):
    row: int
    probability: float
    prediction: int


class BatchRequest(BaseModel):
    instances: list[Patient] = Field(min_length=1, max_length=1000)


def _require_ready() -> None:
    if "production_model" not in state:
        raise HTTPException(
            status_code=503,
            detail=state.get("error", "model artifacts not loaded; run `kedro run`"),
        )


def _score(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The inference pipeline, in-process: identical functions, identical order."""
    params = state["params"]
    data = clean_data(frame, params["columns"])
    data = apply_imputer(data, state["imputer"])
    data = apply_outlier_bounds(data, state["outlier_bounds"])
    data = engineer_features(data, params["feature_engineering"])
    data = apply_encoder(data, state["encoder"])
    data = apply_scaler(data, state["scaler"])
    return predict_node(state["production_model"], data, params["inference"])


@app.get("/", tags=["operations"])
def root() -> dict[str, Any]:
    return {"service": app.title, "version": app.version, "docs": "/docs"}


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready", tags=["operations"])
def ready() -> dict[str, Any]:
    """Readiness: the model artifacts are loaded and scoring can serve traffic."""
    _require_ready()
    model = state["production_model"]
    return {
        "status": "ready",
        "model": model["name"],
        "n_features": len(model["feature_columns"]),
        "selection": model.get("selection"),
    }


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict(patient: Patient) -> dict[str, Any]:
    _require_ready()
    try:
        return _score(pd.DataFrame([patient.model_dump()]))[0]
    except Exception as exc:
        logger.exception("scoring failed")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.post("/predict/batch", response_model=list[Prediction], tags=["inference"])
def predict_batch(request: BatchRequest) -> list[dict[str, Any]]:
    _require_ready()
    try:
        return _score(pd.DataFrame([p.model_dump() for p in request.instances]))
    except Exception as exc:
        logger.exception("batch scoring failed")
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.get("/datasets", tags=["datasets"])
def list_datasets() -> dict[str, list[str]]:
    """Kedro catalog datasets exposed over HTTP."""
    return {"datasets": list(EXPOSED_DATASETS)}


@app.get("/datasets/{name}", tags=["datasets"])
def read_dataset(
    name: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Serve a catalog dataset as JSON, paginated."""
    if name not in EXPOSED_DATASETS:
        raise HTTPException(404, f"unknown dataset '{name}'; see GET /datasets")
    if "catalog" not in state:
        raise HTTPException(503, state.get("error", "catalog unavailable"))
    try:
        data = state["catalog"].load(name)
    except Exception as exc:
        raise HTTPException(
            404, f"dataset '{name}' has not been produced yet: {exc}"
        ) from exc

    records = data.to_dict("records") if isinstance(data, pd.DataFrame) else data
    return {
        "dataset": name,
        "total": len(records),
        "offset": offset,
        "limit": limit,
        "records": records[offset : offset + limit],
    }
