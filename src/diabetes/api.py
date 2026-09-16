"""FastAPI application that serves the Kedro-trained diabetes model.

The Kedro project is bootstrapped exactly once per process, and the fitted
artifacts are read from the catalog exactly once. Scoring then calls the same
node functions the batch pipeline uses, so an HTTP prediction and a
``kedro run --pipeline inference`` prediction cannot diverge.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from kedro.framework.project import configure_project
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

PACKAGE_NAME = "diabetes"
PROJECT_PATH = Path(__file__).resolve().parents[2]
ARTIFACTS = ("production_model", "imputer", "outlier_bounds", "encoder", "scaler")
EXPOSED_DATASETS = (
    "raw_modelling_data",
    "raw_inference_data",
    "master_table",
    "inference_predictions",
)


# ---------------------------------------------------------------------------
# One-time Kedro bootstrap and artifact load
# ---------------------------------------------------------------------------

_bootstrap_lock = threading.Lock()
_bootstrapped = False

_state_lock = threading.Lock()
state: dict[str, Any] = {}


def _ensure_bootstrap() -> None:
    """Bootstrap the Kedro project exactly once (thread-safe)."""
    global _bootstrapped  # noqa: PLW0603
    if _bootstrapped:
        return
    with _bootstrap_lock:
        if not _bootstrapped:
            bootstrap_project(PROJECT_PATH)
            configure_project(PACKAGE_NAME)
            _bootstrapped = True
            logger.info("bootstrapped Kedro project at %s", PROJECT_PATH)


def _ensure_state() -> None:
    """Read parameters and fitted artifacts from the catalog exactly once.

    A single ``KedroSession`` is opened here and never again: the catalog and the
    unpickled artifacts are held in memory for the lifetime of the process, so no
    request pays for a session, a config load or a disk read of the model.
    """
    if state:
        return
    with _state_lock:
        if state:
            return
        _ensure_bootstrap()
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            context = session.load_context()
            loaded = {name: context.catalog.load(name) for name in ARTIFACTS}
            state.update(loaded, catalog=context.catalog, params=context.params)
        logger.info("loaded artifacts: %s", ", ".join(ARTIFACTS))


def _require_state() -> dict[str, Any]:
    """Return the loaded state, or 503 with the reason it is unavailable.

    Loading is retried here rather than only at startup, so an API started
    before the first ``kedro run`` recovers without a restart.
    """
    try:
        _ensure_state()
    except Exception as exc:
        # The cause (paths, stack) is logged, not returned: a 503 body is not
        # the place to disclose the server's filesystem layout.
        logger.error("artifacts unavailable — run `kedro run` first: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="model artifacts are not loaded; run `kedro run` first",
        ) from exc
    return state


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class Patient(BaseModel):
    """Raw measurements, exactly as they appear in the source CSV.

    ``0`` is this dataset's sentinel for "not measured" on Glucose,
    BloodPressure, SkinThickness, Insulin and BMI, so it stays a legal value and
    is imputed by the same artifact the training pipeline fitted.
    """

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


class ReadyResponse(BaseModel):
    status: str
    model: str
    n_features: int
    selection: dict[str, Any] | None = None


class DatasetListResponse(BaseModel):
    datasets: list[str]


class DatasetResponse(BaseModel):
    dataset: str
    total: int
    offset: int
    limit: int
    records: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The inference pipeline, in-process: same functions, same order.

    Blocking and CPU-bound, so every endpoint calls it off the event loop.
    """
    params = state["params"]
    data = clean_data(frame, params["columns"])
    data = apply_imputer(data, state["imputer"])
    data = apply_outlier_bounds(data, state["outlier_bounds"])
    data = engineer_features(data, params["feature_engineering"])
    data = apply_encoder(data, state["encoder"])
    data = apply_scaler(data, state["scaler"])
    return predict_node(state["production_model"], data, params["inference"])


async def _score_async(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Run the scoring chain in a worker thread, keeping the loop responsive."""
    try:
        return await run_in_threadpool(_score, frame)
    except Exception as exc:
        logger.exception("scoring failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pay the bootstrap and unpickling cost at startup, not on the first request.
    # A failure here is not fatal: /ready reports 503 and retries the load.
    try:
        await run_in_threadpool(_ensure_state)
    except Exception as exc:
        logger.error("startup load failed — run `kedro run` first: %s", exc)
    yield
    state.clear()


app = FastAPI(
    title="Diabetes Prediction API",
    description="Serves the Kedro-trained diabetes incidence model.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- Operations -----------------------------------------------------------


@app.get("/", tags=["operations"])
def root() -> dict[str, Any]:
    """Service banner."""
    return {"service": app.title, "version": app.version, "docs": "/docs"}


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Liveness: the process is up. Never touches the model."""
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse, tags=["operations"])
def ready() -> ReadyResponse:
    """Readiness: artifacts are loaded and scoring can serve traffic."""
    model = _require_state()["production_model"]
    return ReadyResponse(
        status="ready",
        model=model["name"],
        n_features=len(model["feature_columns"]),
        selection=model.get("selection"),
    )


# ---- Inference ------------------------------------------------------------


@app.post("/predict", response_model=Prediction, tags=["inference"])
async def predict(patient: Patient) -> Prediction:
    """Score a single patient from the eight raw measurements."""
    _require_state()
    predictions = await _score_async(pd.DataFrame([patient.model_dump()]))
    return Prediction(**predictions[0])


@app.post("/predict/batch", response_model=list[Prediction], tags=["inference"])
async def predict_batch(request: BatchRequest) -> list[Prediction]:
    """Score up to 1000 patients in one request."""
    _require_state()
    frame = pd.DataFrame([patient.model_dump() for patient in request.instances])
    return [Prediction(**row) for row in await _score_async(frame)]


# ---- Datasets -------------------------------------------------------------


@app.get("/datasets", response_model=DatasetListResponse, tags=["datasets"])
def list_datasets() -> DatasetListResponse:
    """List the Kedro catalog datasets exposed over HTTP."""
    return DatasetListResponse(datasets=list(EXPOSED_DATASETS))


@app.get("/datasets/{name}", response_model=DatasetResponse, tags=["datasets"])
def read_dataset(
    name: str,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> DatasetResponse:
    """Serve a catalog dataset as paginated JSON."""
    if name not in EXPOSED_DATASETS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown dataset '{name}'; see GET /datasets",
        )
    catalog = _require_state()["catalog"]
    try:
        data = catalog.load(name)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=f"dataset '{name}' has not been produced yet: {exc}",
        ) from exc

    records = data.to_dict("records") if isinstance(data, pd.DataFrame) else data
    return DatasetResponse(
        dataset=name,
        total=len(records),
        offset=offset,
        limit=limit,
        records=records[offset : offset + limit],
    )
