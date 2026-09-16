import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from diabetes import api
from diabetes.api import app

PATIENT = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_always_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_the_promoted_model(client):
    response = client.get("/ready")
    assert response.status_code == 200, "run `kedro run` before the API tests"
    assert response.json()["n_features"] > 0


def test_predict_returns_a_calibrated_probability(client):
    body = client.post("/predict", json=PATIENT).json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_an_unknown_field(client):
    assert (
        client.post("/predict", json={**PATIENT, "Cholesterol": 1}).status_code == 422
    )


def test_predict_rejects_an_out_of_range_value(client):
    assert client.post("/predict", json={**PATIENT, "Age": 5}).status_code == 422


def test_batch_scores_every_instance(client):
    response = client.post("/predict/batch", json={"instances": [PATIENT, PATIENT]})
    assert len(response.json()) == 2


def test_dataset_endpoint_paginates(client):
    body = client.get("/datasets/raw_modelling_data?limit=5").json()
    assert len(body["records"]) == 5
    assert body["total"] == 652


def test_unknown_dataset_is_404(client):
    assert client.get("/datasets/secrets").status_code == 404


# ---- Startup cost ---------------------------------------------------------


def test_bootstrap_runs_exactly_once(monkeypatch):
    """`bootstrap_project` re-reads pyproject and re-configures the project, so
    it must never run per request."""
    calls = []
    monkeypatch.setattr(api, "bootstrap_project", calls.append)
    monkeypatch.setattr(api, "configure_project", lambda name: None)
    monkeypatch.setattr(api, "_bootstrapped", False)

    for _ in range(3):
        api._ensure_bootstrap()

    assert len(calls) == 1


def test_requests_never_open_a_kedro_session(client, monkeypatch):
    """Artifacts are unpickled once at startup; serving must not touch Kedro."""
    opened = []
    monkeypatch.setattr(
        api.KedroSession, "create", lambda **kwargs: opened.append(kwargs)
    )

    for _ in range(3):
        assert client.post("/predict", json=PATIENT).status_code == 200
    client.get("/ready")
    client.get("/datasets")

    assert opened == []


def test_post_handlers_are_coroutines():
    """POST endpoints are async and hand the blocking scoring chain to a worker
    thread, so a slow prediction cannot stall the event loop."""
    assert inspect.iscoroutinefunction(api.predict)
    assert inspect.iscoroutinefunction(api.predict_batch)


# ---- Parity with the batch pipeline ---------------------------------------


def test_api_matches_the_batch_pipeline(client):
    """The strongest guarantee in the project: scoring the same rows over HTTP
    and through `kedro run --pipeline inference` must agree to the digit."""
    project = Path(__file__).resolve().parents[1]
    batch_path = project / "data" / "07_model_output" / "inference_predictions.json"
    if not batch_path.exists():
        pytest.skip("run `kedro run` first to produce batch predictions")

    batch = json.loads(batch_path.read_text())[:20]
    raw = pd.read_csv(project / "data" / "01_raw" / "diabetes-dataset-inference.csv")
    payload = raw.drop(columns=["Outcome"]).head(20).to_dict("records")

    response = client.post("/predict/batch", json={"instances": payload})
    assert response.status_code == 200

    for over_http, from_batch in zip(response.json(), batch, strict=True):
        assert over_http["probability"] == pytest.approx(from_batch["probability"])
        assert over_http["prediction"] == from_batch["prediction"]
