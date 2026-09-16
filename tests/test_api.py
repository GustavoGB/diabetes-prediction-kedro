import pytest
from fastapi.testclient import TestClient

from diabetes.api import app

PATIENT = {
    "Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
    "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50,
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
    assert client.post("/predict", json={**PATIENT, "Cholesterol": 1}).status_code == 422


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
