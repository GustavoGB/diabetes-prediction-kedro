# Diabetes Prediction — Kedro

[![CI](https://github.com/GustavoGB/diabetes-prediction-kedro/actions/workflows/ci.yml/badge.svg)](https://github.com/GustavoGB/diabetes-prediction-kedro/actions/workflows/ci.yml)

Production port of `diabetes-prediction.ipynb` into three Kedro pipelines
(**data engineering → modelling → inference**), served by a FastAPI application
and packaged as a self-contained Docker image.

Dataset: 768 Pima records, supplied as 652 modelling rows and 116 held-out
inference rows. Target `Outcome` is imbalanced (~35% positive).

```
data/01_raw/*.csv ──► data_engineering ──► master_table ──► modelling ──► production_model
                            │                                                    │
                            └──► imputer, outlier_bounds, encoder, scaler ────────┤
                                                                                  ▼
                                                                            inference
                                                                          (batch or HTTP)
```

The DAGs are drawn in **[docs/pipelines.md](docs/pipelines.md)** — GitHub renders
them inline, so you can read the graph without starting anything.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) ≥ 0.12 (it downloads Python itself —
no separate install) and, for the container, Docker ≥ 24.

```bash
git clone https://github.com/GustavoGB/diabetes-prediction-kedro.git
cd diabetes-prediction-kedro
uv sync --extra dev          # exact versions from the committed uv.lock
uv run kedro run             # 27 nodes, ~5 seconds
```

Both datasets are committed under `data/01_raw/`, so there is nothing to
download. Prefix commands with `uv run`, or activate once with
`source .venv/bin/activate`.

That single `kedro run` writes:

| output | what it is |
|---|---|
| `data/05_model_input/master_table.csv` | 652 × 35 model-ready table |
| `data/06_models/production_model.pkl/<version>/` | the promoted model |
| `data/06_models/{imputer,encoder,scaler}.pkl/<version>/` | fitted transformers |
| `data/06_models/outlier_bounds.json/<version>/` | per-column Tukey fences |
| `data/07_model_output/inference_predictions.json` | 116 scored rows |
| `data/08_reporting/*_metrics.json` | per-split metrics for both candidates |
| `data/08_reporting/*_quality.json` | Great Expectations reports, three checkpoints |

### One pipeline at a time

```bash
uv run kedro run --pipeline data_engineering   # raw CSV -> master table (13 nodes)
uv run kedro run --pipeline modelling          # train, evaluate, promote (5 nodes)
uv run kedro run --pipeline inference          # score the holdout CSV (9 nodes)
uv run kedro run --pipeline training           # data_engineering + modelling
```

On a clean checkout the order matters — `modelling` needs `master_table`, and
`inference` needs `production_model` plus the four fitted artifacts. Plain
`kedro run` handles that for you. `make help` lists every shortcut.

---

## Visualise it

```bash
uv run kedro viz          # or: make viz  →  http://localhost:4141
```

The pipeline selector (top left) switches between `__default__`,
`data_engineering`, `modelling`, `inference` and `training`. Data datasets are
tagged with their layer — raw, intermediate, primary, feature, model_input,
models, model_output, reporting — so the graph groups itself into bands.

Fitted transformers deliberately carry *no* layer: the encoder is fitted **from**
feature data and then used **to produce** it, so labelling them made the layer
graph cyclic, and kedro-viz answers a cycle by disabling the bands for the whole
graph. A test now fails if that cycle comes back.

No browser? `uv run kedro viz run --no-browser --port 4141`, or just read
[docs/pipelines.md](docs/pipelines.md), which is generated from the same
pipeline definitions by `make diagram`.

---

## Serve it

```bash
uv run uvicorn diabetes.api:app --port 8000     # or: make api
```

```bash
# is a model loaded?
curl -s localhost:8000/ready
# {"status":"ready","model":"LogisticRegression","n_features":33,
#  "selection":{"metric":"roc_auc","split":"test","score":0.8165011982197877}}

# high risk: elevated glucose, obese BMI, strong family history
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
  "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 55}'
# {"row":0,"probability":0.968447,"prediction":1}

# the raw data, over HTTP
curl -s 'localhost:8000/datasets/raw_modelling_data?limit=1'
```

Full surface — batch scoring, validation errors, the dataset endpoints, every
`curl` with its exact response — in **[docs/api.md](docs/api.md)**.

Scoring calls the same node functions the batch pipeline uses, and a test asserts
the two agree to the digit.

---

## Containerise it

```bash
docker compose up --build        # then: http://localhost:8000/docs
```

The image **trains during the build**, so it is self-contained: no model pickles
in git, no volumes to mount, and the artifacts provably match the code in the
image. Retraining means rebuilding. First build ~2 minutes.

---

## Results

Held-out numbers, stratified 70/30 split, `random_state=17`:

| model | split | accuracy | precision | recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| LogisticRegression (baseline) | test | 0.719 | 0.597 | 0.623 | 0.610 | **0.817** |
| RandomForest (tuned, 27-point grid) | test | 0.714 | 0.589 | 0.623 | 0.606 | 0.811 |

`select_best_model` promoted the **baseline LogisticRegression** — the tuned
forest wins cross-validation (0.839) but loses on the held-out split, a textbook
case of a grid overfitting CV folds at n=456.

Scored against the 116 never-seen inference rows: **accuracy 0.733, ROC AUC
0.814, F1 0.627.** Test AUC 0.817 vs holdout AUC 0.814 is the evidence that the
leakage fixes worked — the estimate generalises.

> The notebook reports up to 0.80 accuracy. Those numbers are inflated; see
> [docs/architecture.md](docs/architecture.md#what-changed-from-the-notebook).

---

## Rollback

The five artifacts inference loads — the model and the four transformers that
shaped its features — are versioned **as a set**. Versioning the model alone
would let a rollback pair an old estimator with new features, and that failure is
silent, because the shapes still line up. Kedro stamps one `save_version` per
run, so a run's artifacts share a timestamp:

```bash
make versions                                  # list them, newest last
make rollback VERSION=2026-09-17T01.04.59.781Z # score with an earlier set
```

---

## Tests

```bash
uv run pytest -q          # 59 passed
make lint                 # black --check + ruff
```

The suite targets contracts rather than line coverage: that inference may only
*apply* what training fitted, that HTTP and batch scores agree, that a degraded
feed fails even when every individual row is valid, and that the layer graph
stays acyclic. `tests/test_api.py` needs artifacts on disk — run `kedro run`
first, or those tests report the 503 and the parity test skips.

Every push and pull request runs the same checks plus a full Docker smoke test —
see [docs/ci.md](docs/ci.md).

---

## Where to read more

| document | what is in it |
|---|---|
| [docs/pipelines.md](docs/pipelines.md) | the three DAGs, drawn — generated from the code |
| [docs/architecture.md](docs/architecture.md) | every node, what changed from the notebook, config reference, repo layout |
| [docs/api.md](docs/api.md) | the complete HTTP surface, with real responses |
| [docs/data-quality.md](docs/data-quality.md) | why Pydantic **and** Great Expectations, and what each catches that the other cannot |
| [docs/ci.md](docs/ci.md) | the three CI jobs and what they actually assert |
| [docs/troubleshooting.md](docs/troubleshooting.md) | errors you can hit on a fresh checkout |
