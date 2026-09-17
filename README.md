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

---

# Table of contents

1. [Requirements](#1-requirements)
2. [Install](#2-install)
3. [Run the pipelines](#3-run-the-pipelines)
4. [Visualise the pipelines](#4-visualise-the-pipelines-kedro-viz)
5. [Run the API](#5-run-the-api)
6. [Run with Docker](#6-run-with-docker)
7. [Run the tests](#7-run-the-tests)
8. [Continuous integration](#8-continuous-integration)
9. [Troubleshooting](#9-troubleshooting)
10. [Data quality: Pydantic + Great Expectations](#10-data-quality-pydantic--great-expectations)
11. [How it is built](#11-how-it-is-built)
12. [Results](#12-results)
13. [What changed from the notebook](#13-what-changed-from-the-notebook)
14. [Configuration reference](#14-configuration-reference)
15. [Project layout](#15-project-layout)

---

## 1. Requirements

| tool | version | needed for |
|---|---|---|
| Python | 3.10 – 3.13 | everything (3.12 recommended) |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.12 | dependency management |
| Docker | ≥ 24 | the container only |
| make | any | the shortcut targets only |

Install uv if you do not have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

No separate Python install is needed — uv downloads the interpreter itself.

---

## 2. Install

```bash
git clone https://github.com/GustavoGB/diabetes-prediction-kedro.git
cd diabetes-prediction-kedro
uv sync --extra dev
```

`uv sync` creates `.venv/` and installs the **exact** versions pinned in the
committed `uv.lock`. `--extra dev` adds pytest, black and ruff; drop it for a
runtime-only install.

Both datasets are committed under `data/01_raw/`, so there is nothing to
download.

Prefix commands with `uv run` (no manual venv activation needed), or activate
once with `source .venv/bin/activate`.

---

## 3. Run the pipelines

### Everything at once

```bash
uv run kedro run
```

Runs **27 nodes** — data engineering, modelling and inference — in about 5
seconds, and writes:

| output | what it is |
|---|---|
| `data/05_model_input/master_table.csv` | 652 × 35 model-ready table |
| `data/06_models/imputer.pkl` | fitted RobustScaler + KNNImputer |
| `data/06_models/outlier_bounds.json` | per-column Tukey fences |
| `data/06_models/encoder.pkl` | fitted OneHotEncoder |
| `data/06_models/scaler.pkl` | fitted RobustScaler |
| `data/06_models/baseline_model.pkl`, `tuned_model.pkl` | the two candidates |
| `data/06_models/production_model.pkl` | the promoted model |
| `data/08_reporting/*_metrics.json` | per-split metrics for both candidates |
| `data/07_model_output/inference_predictions.json` | 116 scored rows |
| `data/08_reporting/*_quality.json` | Great Expectations reports for the three data quality checkpoints |

### One pipeline at a time

```bash
uv run kedro run --pipeline data_engineering   # raw CSV -> master table (13 nodes)
uv run kedro run --pipeline modelling          # train, evaluate, promote (5 nodes)
uv run kedro run --pipeline inference          # score the holdout CSV (9 nodes)
uv run kedro run --pipeline training           # data_engineering + modelling
```

Order matters on a clean checkout: `modelling` needs `master_table`, and
`inference` needs `production_model` plus the four fitted artifacts. Running
plain `kedro run` handles the ordering for you.

### Useful flags

```bash
# run a single node
uv run kedro run --pipeline modelling --nodes train_baseline_model

# resume partway through a pipeline
uv run kedro run --pipeline data_engineering --from-nodes engineer_features

# override any parameter from the command line
uv run kedro run --pipeline inference --params inference.threshold=0.35
```

### Make shortcuts

```bash
make help       # list every target
make train      # kedro run --pipeline training
make infer      # kedro run --pipeline inference
```

---

## 4. Visualise the pipelines (kedro viz)

```bash
uv run kedro viz          # or: make viz
```

Opens `http://localhost:4141`. Use the pipeline selector (top left) to switch
between `__default__`, `data_engineering`, `modelling`, `inference` and
`training`. Datasets are tagged with their layer — raw, intermediate, primary,
feature, model_input, models, model_output, reporting — so the graph groups
itself.

If the browser does not open automatically:

```bash
uv run kedro viz run --no-browser --port 4141
```

---

## 5. Run the API

Train first — the API serves artifacts, it does not create them:

```bash
uv run kedro run --pipeline training
uv run uvicorn diabetes.api:app --reload --port 8000   # or: make api
```

Interactive docs: **http://localhost:8000/docs**

| method | route | purpose |
|---|---|---|
| GET | `/` | service banner |
| GET | `/health` | liveness — the process is up |
| GET | `/ready` | readiness — artifacts loaded (503 before the first `kedro run`) |
| POST | `/predict` | score one patient |
| POST | `/predict/batch` | score up to 1000 patients |
| GET | `/datasets` | list the catalog datasets exposed over HTTP |
| GET | `/datasets/{name}` | read a dataset as JSON (`?limit=&offset=`) |

### 5.1 Operations

```bash
curl localhost:8000/
# {"service":"Diabetes Prediction API","version":"0.1.0","docs":"/docs"}

curl localhost:8000/health
# {"status":"ok"}

curl localhost:8000/ready
# {"status":"ready","model":"LogisticRegression","n_features":33,
#  "selection":{"metric":"roc_auc","split":"test","score":0.8165011982197877}}
```

Before the first `kedro run` there is no model to serve. `/health` still answers
200 (the process is alive), while `/ready` and the scoring routes answer **503**:

```bash
curl -w ' [%{http_code}]' localhost:8000/ready
# {"detail":"model artifacts are not loaded; run `kedro run` first"} [503]
```

Train in another terminal and the **same process recovers on the next request** —
no restart:

```bash
uv run kedro run --pipeline training
curl localhost:8000/ready
# {"status":"ready","model":"LogisticRegression",...}
```

### 5.2 Single predictions

```bash
# high risk: elevated glucose, obese BMI, strong family history
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
  "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 55}'
# {"row":0,"probability":0.968447,"prediction":1}

# low risk: normal glucose, healthy BMI, young
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
  "Insulin": 94, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31}'
# {"row":0,"probability":0.048499,"prediction":0}

# borderline: below the 0.5 threshold, but not by much
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 3, "Glucose": 129, "BloodPressure": 74, "SkinThickness": 26,
  "Insulin": 205, "BMI": 33.2, "DiabetesPedigreeFunction": 0.591, "Age": 25}'
# {"row":0,"probability":0.32855,"prediction":0}

# unmeasured fields: 0 is the sentinel, and the fitted KNN imputer fills it in
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 8, "Glucose": 125, "BloodPressure": 96, "SkinThickness": 0,
  "Insulin": 0, "BMI": 0, "DiabetesPedigreeFunction": 0.232, "Age": 54}'
# {"row":0,"probability":0.473659,"prediction":0}
```

### 5.3 Batch predictions

```bash
curl -X POST localhost:8000/predict/batch -H 'content-type: application/json' -d '{
  "instances": [
    {"Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
     "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 55},
    {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
     "Insulin": 94, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 31},
    {"Pregnancies": 0, "Glucose": 137, "BloodPressure": 40, "SkinThickness": 35,
     "Insulin": 168, "BMI": 43.1, "DiabetesPedigreeFunction": 2.288, "Age": 33}]}'
# [{"row":0,"probability":0.968447,"prediction":1},
#  {"row":1,"probability":0.048499,"prediction":0},
#  {"row":2,"probability":0.904905,"prediction":1}]
```

`row` is the index of the instance in your request, so responses map back to
your input order.

Score the **entire holdout CSV** through the API and compare it with the batch
pipeline — both report 43 positives, because they run the same code:

```bash
python3 -c "
import csv, json
rows = list(csv.DictReader(open('data/01_raw/diabetes-dataset-inference.csv')))
floats = {'BMI', 'DiabetesPedigreeFunction'}
payload = [{k: (float(v) if k in floats else int(v))
            for k, v in r.items() if k != 'Outcome'} for r in rows]
json.dump({'instances': payload}, open('/tmp/payload.json', 'w'))"

curl -s -X POST localhost:8000/predict/batch \
  -H 'content-type: application/json' -d @/tmp/payload.json \
  | python3 -c "import json,sys; d=json.load(sys.stdin); \
      print('scored:', len(d), '| positives:', sum(x['prediction'] for x in d))"
# scored: 116 | positives: 43
```

### 5.4 Validation errors

Every payload is validated by Pydantic **before** any model code runs, so a bad
request is a fast 422 rather than a wrong answer.

```bash
# unknown field -> 422 (extra="forbid" catches typos and stray columns)
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
  "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 55,
  "Cholesterol": 210}'
# {"detail":[{"type":"extra_forbidden","loc":["body","Cholesterol"],
#             "msg":"Extra inputs are not permitted","input":210}]}

# out of range -> 422
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
  "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 5}'
# {"detail":[{"type":"greater_than_equal","loc":["body","Age"],
#             "msg":"Input should be greater than or equal to 21","input":5,
#             "ctx":{"ge":21}}]}

# missing fields -> 422, listing every one of them at once
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"Glucose": 183, "Age": 55}'
# {"detail":[{"type":"missing","loc":["body","Pregnancies"],"msg":"Field required",...},
#            ... one entry per missing field ...]}

# empty batch -> 422
curl -X POST localhost:8000/predict/batch -H 'content-type: application/json' \
  -d '{"instances": []}'
# {"detail":[{"type":"too_short","loc":["body","instances"],
#             "msg":"List should have at least 1 item after validation, not 0",...}]}
```

### 5.5 Datasets over HTTP

```bash
curl localhost:8000/datasets
# {"datasets":["raw_modelling_data","raw_inference_data","master_table",
#              "inference_predictions"]}

# the source data, one row
curl 'localhost:8000/datasets/raw_modelling_data?limit=1'
# {"dataset":"raw_modelling_data","total":652,"offset":0,"limit":1,
#  "records":[{"Pregnancies":3,"Glucose":162,"BloodPressure":52,"SkinThickness":38,
#              "Insulin":0,"BMI":37.2,"DiabetesPedigreeFunction":0.652,"Age":24,
#              "Outcome":1}]}

# paginate the batch predictions
curl 'localhost:8000/datasets/inference_predictions?limit=3&offset=10'
# {"dataset":"inference_predictions","total":116,"offset":10,"limit":3,
#  "records":[{"row":10,"probability":0.313068,"prediction":0},
#             {"row":11,"probability":0.898523,"prediction":1},
#             {"row":12,"probability":0.477521,"prediction":0}]}

# the engineered feature table the model actually trains on
curl 'localhost:8000/datasets/master_table?limit=1'

# anything not on the allow-list -> 404
curl localhost:8000/datasets/secrets
# {"detail":"unknown dataset 'secrets'; see GET /datasets"}
```

Only the four datasets in `EXPOSED_DATASETS` are reachable; the path is checked
against that tuple, so no catalog entry can be read by guessing its name.

### 5.6 Handy one-liners

```bash
# the OpenAPI schema, generated from the Pydantic models
curl -s localhost:8000/openapi.json | python3 -m json.tool | head -40

# how long a prediction takes end to end
curl -s -o /dev/null -w 'predict: %{time_total}s\n' \
  -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
  "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 55}'
# predict: 0.013922s

# just the status code
curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/ready

# pretty-print any response without jq
curl -s localhost:8000/ready | python3 -m json.tool

# 20 concurrent predictions
for i in $(seq 1 20); do
  curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
    "Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
    "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 55}' &
done; wait
```

If you have `jq`, swap `python3 -m json.tool` for `jq .` anywhere above.

### Request contract

Send the **8 raw measurements**; the service runs the full transform chain
server-side, so clients never handle the 33 engineered features.

| field | type | range | notes |
|---|---|---|---|
| `Pregnancies` | int | 0–20 | `0` is a real value |
| `Glucose` | int | 0–300 | `0` = not measured |
| `BloodPressure` | int | 0–200 | `0` = not measured |
| `SkinThickness` | int | 0–110 | `0` = not measured |
| `Insulin` | int | 0–1000 | `0` = not measured |
| `BMI` | float | 0–80 | `0` = not measured |
| `DiabetesPedigreeFunction` | float | >0–3 | |
| `Age` | int | 21–120 | |

#### Validation with Pydantic

The contract above is not documentation — it is enforced. `src/diabetes/api.py`
declares six Pydantic models and every route is bound to one:

| model | used by | what it enforces |
|---|---|---|
| `Patient` | `POST /predict` | the 8 fields, their types, `Field(ge=…, le=…)` bounds, and `model_config = {"extra": "forbid"}` |
| `BatchRequest` | `POST /predict/batch` | `list[Patient]` with `min_length=1, max_length=1000` |
| `Prediction` | both scoring routes | the `{row, probability, prediction}` response shape |
| `ReadyResponse` | `GET /ready` | the readiness payload |
| `DatasetListResponse` | `GET /datasets` | the dataset allow-list |
| `DatasetResponse` | `GET /datasets/{name}` | the paginated envelope |

Consequences worth knowing:

- A misspelled or extra field is a **422**, not a silently ignored key — without
  `extra="forbid"`, `{"Glucoze": 183}` would quietly score a patient whose
  glucose was never supplied.
- Validation runs **before** any pandas or scikit-learn code, so malformed input
  can never reach the transform chain.
- `response_model=` on every route means the response shape is validated on the
  way out too — a node that started returning a different key would fail loudly.
- `/docs` and `/openapi.json` are generated from these models, so the API
  documentation cannot drift from what the code accepts.

---

## 6. Run with Docker

```bash
docker compose up --build        # or: make docker-build && make docker-up
```

Then open http://localhost:8000/docs. Stop with `docker compose down`.

Without compose:

```bash
docker build -t diabetes-api .
docker run --rm -p 8000:8000 diabetes-api
```

The image **trains during the build** (`kedro run --pipeline training`), so it is
self-contained: no model pickles in git, no volumes to mount, and the artifacts
provably match the code in the image. Retraining means rebuilding. First build
takes ~2 minutes; the dependency layer is cached afterwards.

---

## 7. Run the tests

```bash
uv run pytest -q          # 56 passed
uv run black src tests scripts    # format   (make format)
uv run black --check src tests scripts && uv run ruff check src tests scripts   # (make lint)
```

`tests/test_api.py` needs artifacts on disk — run `kedro run` first, or those
tests will report the 503 and the parity test will skip.

| file | tests | covers |
|---|---|---|
| `test_data_engineering.py` | 11 | sentinel zeros, missing target/columns, fit-nodes never see test rows, encoder schema stability, unseen categories are logged, values above the top bin still get a band, outlier clipping, stratification |
| `test_modelling.py` | 4 | the feature contract, config-driven estimator swap, per-split metrics, promotion by holdout score |
| `test_inference.py` | 6 | the DataFrame/JSON adapter, threshold behaviour, feature selection, row-index preservation |
| `test_pipelines.py` | 12 | registry names, unique node names, inference reuses the *same* transform functions, inference never fits, every artifact is persisted and reused, no orphan catalog entries, viz layers, both branches are validated, validation actually gates |
| `test_validation.py` | 9 | a healthy batch passes untouched, out-of-range values fail, a degraded feed fails *even though every row is valid*, class-imbalance drift fails, truncation fails, advisory mode, absent columns skipped, **the shipped suite accepts an unlabelled batch** |
| `test_api.py` | 14 | health/ready, validation, batch, dataset endpoints, bootstrap idempotence, no session per request, async handlers, 503 *and* 404 without leaking paths, **API-vs-batch parity** |

The suite targets contracts rather than line coverage. The two that matter most:

- `test_inference_never_fits_an_artifact` — scoring may only *apply* what
  training fitted.
- `test_api_matches_the_batch_pipeline` — the same rows scored over HTTP and via
  `kedro run --pipeline inference` must agree to the digit.
- `test_a_degraded_feed_fails_even_though_every_row_is_valid` — the case Pydantic
  structurally cannot see (see [section 10](#10-data-quality-pydantic--great-expectations)).

---

## 8. Continuous integration

Every push to `main` and every pull request targeting `main` runs
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) on GitHub Actions. Three
jobs run in parallel, on a clean `ubuntu-latest` checkout with no cached
artifacts — so CI proves the repository is reproducible from nothing but what is
committed.

| job | what it does | why it is separate |
|---|---|---|
| **Format, lint, tests** | `black --check`, `ruff check`, `kedro run`, then `pytest -q` (56 tests) | fastest feedback; a formatting slip should not wait on Docker |
| **Pipeline end to end** | `kedro run`, then asserts the *outputs* on disk | exit code 0 only proves Kedro did not raise — see below |
| **Image builds and serves** | `docker build`, start the container, poll `/ready`, score a patient over HTTP | the only job that exercises the shipped artefact, not the source tree |

`concurrency` cancels a superseded run: pushing twice to the same branch kills
the first run rather than letting both compete for runners.

### Why the pipeline job asserts outputs

A green `kedro run` means no node raised an exception. It does not mean the run
produced anything usable, so the job then checks, in a plain Python step:

- `data/06_models/production_model.pkl` exists and is non-empty — a model was
  actually promoted;
- all three `data/08_reporting/*_quality.json` suites report `"success": true` —
  the Great Expectations gates passed (see
  [section 10](#10-data-quality-pydantic--great-expectations));
- `inference_predictions.json` holds exactly **116** rows — the full held-out
  batch was scored, not a truncated slice;
- every probability is in `[0, 1]` and every prediction is `0` or `1`.

### Why the Docker smoke test asserts a band, not a number

The image retrains on build, then promotes one of two candidates by test
`roc_auc`: LogisticRegression (0.8165) against a tuned RandomForest (0.8115) — a
margin of 0.005. KNN imputation and a 27-candidate `GridSearchCV` are sensitive
to the BLAS and scikit-learn build, so a different runner can flip the winner,
and the same high-risk patient then scores 0.885 instead of 0.968. The test
therefore asserts `|p - 0.968447| <= 0.15`: wide enough to cover either
candidate, tight enough to fail if the feature order, the scaler or the model
regresses. Pinning the exact float would produce a flaky build, not a stronger
check.

### Running the same checks locally

CI runs nothing you cannot run yourself — this is the whole workflow, in order:

```bash
uv sync --locked --extra dev                      # fails if uv.lock is out of date
uv run black --check src tests scripts
uv run ruff check src tests scripts
uv run kedro run                                  # must precede pytest
uv run pytest -q
docker compose build && docker compose up -d      # only if the image changed
curl -fsS localhost:8000/ready
```

`kedro run` has to come before `pytest`: `tests/test_api.py` drives the real
FastAPI app, which loads the fitted artifacts from `data/06_models` — gitignored,
so a fresh checkout has none and the API tests would report 503. The *default*
pipeline is required rather than `--pipeline training`, because the parity test
compares HTTP scores against `data/07_model_output/inference_predictions.json`
and silently skips if the inference pipeline never ran.

### Pinned versions

Both actions are pinned to a major tag that resolves — `actions/checkout@v5`
and `astral-sh/setup-uv@v7`. Neither is the newest release (checkout is on v7,
setup-uv on 10.x); they are pinned because they work, so a major bump is a
deliberate PR that CI can vet, not a silent upgrade. Note that astral-sh stopped
publishing bare major tags after `v7`, so `@v10` would not resolve at all.
`uv sync --locked` asserts `uv.lock` is current: a dependency added to
`pyproject.toml` without re-running `uv lock` fails CI instead of installing a
stale lock behind your back. CI installs Python **3.12** to match the `python:3.12-slim`
base image in the [Dockerfile](Dockerfile), so the tested interpreter is the
shipped one. The lockfile covers `>=3.10,<3.14`, so developing on 3.11 locally is
fine.

### Pull requests

[`.github/pull_request_template.md`](.github/pull_request_template.md) pre-fills
every PR with the checklist above, so the manual steps CI cannot judge — "is the
README still true?" — are not forgotten.

To make the checks *mandatory* rather than advisory, protect `main` so a branch
cannot merge until all three jobs are green. In the GitHub UI: **Settings →
Branches → Add branch ruleset → Require status checks to pass**, then select the
three checks by their job names:

- `Format, lint, tests`
- `Pipeline end to end`
- `Image builds and serves`

The equivalent `gh api -X PUT .../branches/main/protection` call exists, but the
REST body is fussy about types and required keys — use the UI unless you are
scripting it.

---

## 9. Troubleshooting

**`{"detail":"model artifacts are not loaded; run `kedro run` first"}` with HTTP 503**
The model has not been trained yet. Run `uv run kedro run --pipeline training`.
The API retries the load on the next request, so the running process recovers on
its own — no restart needed. The underlying `DatasetError` (with the path it
looked for) is written to the server log, not returned to the client.

**`ValueError: Pipeline contains no nodes` / `Failed to find the pipeline`**
Run from the project root (the directory holding `pyproject.toml`).

**`ModuleNotFoundError: No module named 'diabetes'`**
The project is not installed in the venv. Re-run `uv sync --extra dev`, and
prefix commands with `uv run`.

**`input is missing required feature columns: [...]`**
The CSV or payload lacks one of the eight raw columns. Column names are matched
case-insensitively (everything is upper-cased first), but they must be present.

**Port already in use**
`uv run uvicorn diabetes.api:app --port 8001`, or `kedro viz run --port 4142`.

**Docker build fails at `COPY README.md`**
This file must exist — `pyproject.toml` declares `readme = "README.md"` and
setuptools cannot build the wheel without it.

**Kedro asks about telemetry**
A `.telemetry` file with `consent: false` is committed, so it should not.

---

## 10. Data quality: Pydantic + Great Expectations

The project validates at **two different boundaries**, with two different tools,
because they answer two different questions.

| | Pydantic | Great Expectations |
|---|---|---|
| question | *is this **record** well-formed?* | *is this **batch** fit to train or score on?* |
| unit | one row / one request | a whole dataframe |
| boundary | the HTTP edge (`api.py`) | the pipeline edge (`validation.py`) |
| sees | fields, types, per-field bounds | row counts, null rates, distributions, cardinality, medians |
| when | every request, synchronously | every `kedro run`, per checkpoint |
| failure | **422** to the caller | the run **stops**; a report lands in `08_reporting/` |
| cost | microseconds | ~10 ms per suite on 650 rows |
| declared in | `class Patient(BaseModel)` | `parameters.yml → data_quality` |

### Why neither can do the other's job

This is not a matter of preference — it is a **category difference**.

A Pydantic model validates one object. "48% of `Insulin` is missing", "the
positive class collapsed from 35% to 3%", "the feed returned 5 rows instead of
650" are all properties of a **collection**. There is no field you can annotate
to express them, because no single record is wrong. That is the entire blind
spot, and it is exactly where data pipelines fail in practice: not with garbage
rows, but with perfectly well-formed rows that no longer mean what they used to.

The reverse is just as true. Running a GX suite per HTTP request would be
nonsense: row-count and distribution expectations are undefined on a batch of
one, and you would pay ~10 ms and a validation context for a call that currently
takes 13 ms end to end. Pydantic costs microseconds and returns a precise
422 pointing at the offending field.

**So: Pydantic rejects a bad caller. Great Expectations rejects a bad dataset.**

### Demonstrated

Four realistic corruptions, run through both layers against the real inference
CSV. Reproduce it yourself:

```bash
uv run python scripts/data_quality_demo.py      # or: make quality
```

```
0. The real inference batch (baseline)
  Pydantic : 116/116 rows valid
  GX       : PASS (0/16 expectations failed)

1. Upstream job truncated the feed to 5 rows
  Pydantic : 5/5 rows valid                       <-- every row still perfect
  GX       : FAIL (4/16 expectations failed)
             - expect_table_row_count_to_be_between -> 5
             - expect_column_values_to_not_be_null(SKINTHICKNESS) -> 60.0% unexpected
             - expect_column_values_to_not_be_null(INSULIN) -> 80.0% unexpected
             - expect_column_mean_to_be_between(OUTCOME) -> 0.0

2. Insulin sensor stopped reporting (90% now unmeasured)
  Pydantic : 116/116 rows valid                   <-- 0 is a legal value!
  GX       : FAIL (1/16 expectations failed)
             - expect_column_values_to_not_be_null(INSULIN) -> 94.8% unexpected

3. Label pipeline broke: positives collapse to 2.6%
  Pydantic : 116/116 rows valid                   <-- 0 and 1 are both legal
  GX       : FAIL (1/16 expectations failed)
             - expect_column_mean_to_be_between(OUTCOME) -> 0.0258...

4. BMI arrives in the wrong unit (x10)
  Pydantic : 3/116 rows valid                     <-- here they overlap
  GX       : FAIL (1/16 expectations failed)
             - expect_column_values_to_be_between(BMI) -> 100.0% unexpected
```

Cases 1–3 are the argument. **Every single row is valid**, and the dataset is
ruined. A model trained on case 2 would quietly learn from a feature that is now
95% imputed; a model trained on case 3 would learn to always predict 0 and
report 97% accuracy. Both would pass every unit test in this repo and every
Pydantic check in the API.

Case 4 shows the deliberate overlap: bounds exist in both layers. That is
defence in depth, not duplication — the API stops one bad caller at the door,
the pipeline stops a bad upstream feed before it reaches the model.

### The asymmetry is on purpose

`Patient.Age` is `ge=21`; the GX suite allows `AGE >= 18`. That is not a drift
bug. **The API is strict about what it accepts; the pipeline is tolerant about
what history contains.** New requests must fall inside the range the model was
fitted on (the youngest training patient is 21, so a 19-year-old is an
extrapolation the service should refuse). Historical data is allowed to be
messier than what you accept today, because you cannot go back and re-collect
it. Postel's law, applied to a model boundary.

### What this would have caught in this very project

The notebook analysis turned up defects that dataset-level validation catches
directly:

- **The inference CSV has a different missingness profile** — `Glucose` has zero
  sentinel-zeros there, while the modelling CSV has 5. The notebook derived its
  "which columns are zero-coded" list from `min() == 0`, so it would silently
  treat `Glucose` differently on the two files. A null-rate expectation makes
  that visible instead of silent.
- **Schema drift**: replaying the notebook's `pd.get_dummies` chain on the
  inference CSV yields 24 columns, not 25. `ExpectTableColumnsToMatchSet` on the
  cleaned frame plus the fitted `OneHotEncoder` closes that hole from both ends.
- **Imputation silently not running**: the `master_table` suite asserts zero
  nulls in all five sentinel columns. If the imputer were bypassed, the run
  fails instead of scikit-learn raising something obscure ten nodes later.
- **Stratification regressing**: `ExpectColumnMeanToBeBetween(OUTCOME, 0.2, 0.5)`
  fails if the split stops preserving class balance.

### How it is wired here

Suites live in `conf/base/parameters.yml` as `{type, kwargs}` pairs and are
instantiated by name — the same config-over-code indirection the modelling
pipeline uses for `class_path`. Adding a check is a YAML edit.

```yaml
data_quality:
  cleaned:
    suite: cleaned_data
    fail_on_error: true
    expectations:
      - type: ExpectColumnValuesToNotBeNull
        kwargs: {column: INSULIN, mostly: 0.40}   # ~48% unmeasured is normal
      - type: ExpectColumnMeanToBeBetween
        kwargs: {column: OUTCOME, min_value: 0.20, max_value: 0.50}
```

Three checkpoints, 27 nodes total:

| checkpoint | suite | guards against |
|---|---|---|
| cleaned modelling data | `cleaned_data` | a bad training feed |
| `master_table` | `master_table` | the pipeline's own post-conditions |
| cleaned inference data | `cleaned_data` | **drift** — the same suite as training |

That third row is the one worth pausing on: **the inference batch is held to the
identical contract as the training data**. If next month's batch stops looking
like what the model was fitted on, the run fails loudly rather than producing
confident nonsense.

`validate_data` returns the dataframe *and* a report, so downstream nodes take
the validated frame as input. That dependency is what makes the check a **gate**
rather than a passive observation — a test (`test_validation_gates_downstream_work`)
asserts every validation node's output is actually consumed. Reports are written
to `data/08_reporting/*_quality.json`, and `fail_on_error: false` switches a
checkpoint from blocking to advisory if you would rather score and alert.

### An honest caveat on the choice

Great Expectations is not free: it adds ~60 MB and a sizeable dependency tree
(GX itself, `altair`, `cryptography`) to a project whose core is ~700 lines. For
a codebase this size, [**pandera**](https://pandera.readthedocs.io/) delivers
most of the same value at a fraction of the weight, and its schemas are
declarative classes that sit naturally beside Pydantic models.

GX earns its place here for three reasons: its expectation vocabulary is the
industry reference and reads as documentation; validation results are
first-class artifacts you can persist, diff and alert on; and the expectation
names themselves communicate intent to whoever inherits the pipeline. On a
larger project the calculus is clearer still. On a smaller one, reach for
pandera and keep the idea — the idea is the part that matters.

Two anti-patterns to avoid either way: do not run dataset validation inside the
request path, and do not assert the same bound in five places — pick the
boundary each rule belongs to and keep it there.

---

## 11. How it is built

### `data_engineering` — 13 nodes, raw CSV → master table

```
raw_modelling_data
  → clean_data              upper-case schema, sentinel 0 → NaN
  → validate_data           Great Expectations suite: row count, ranges,
                            null budgets, class balance            [GATE]
  → split_data              stratified train/test tag in a SPLIT column
  → fit_imputer  ─────────┐ RobustScaler + KNNImputer(k=5)       [fit: train only]
  → apply_imputer ────────┘
  → fit_outlier_bounds ───┐ Tukey fences on p05/p95              [fit: train only]
  → apply_outlier_bounds ─┘
  → engineer_features       7 derived columns (stateless)
  → fit_encoder ──────────┐ OneHotEncoder(handle_unknown=ignore) [fit: train only]
  → apply_encoder ────────┘
  → fit_scaler ───────────┐ RobustScaler on 10 numeric columns   [fit: train only]
  → apply_scaler ─────────┘
  → validate_data           post-conditions: no NaN left, balance kept  [GATE]
  → master_table            652 × 35
```

Every stateful step is a **`fit_*` / `apply_*` pair**. The `fit_*` node emits an
artifact the catalog persists in `data/06_models/`; the `apply_*` node consumes
it and never refits. That is the whole design:

- `fit_*` nodes read only rows where `SPLIT == "train"`, so nothing from the
  test split can reach a fitted object. They raise if handed unsplit data.
- The inference pipeline **imports the same `apply_*` function objects** from
  `data_engineering`, so there is exactly one implementation of clean / impute /
  cap / engineer / encode / scale. Training-serving skew is structurally
  impossible rather than merely unlikely, and a test asserts the function
  identity.

### `modelling` — 5 nodes

Trains a baseline and a grid-searched challenger, evaluates both on the held-out
split, and promotes the winner to `production_model`.

The estimator is a `class_path` string in `conf/base/parameters.yml`, so swapping
`RandomForestClassifier` for `HistGradientBoostingClassifier` or
`lightgbm.LGBMClassifier` is a one-line config change with no code edit.

Each model artifact is self-describing — `{estimator, target_column,
feature_columns, eval_splits}` — so inference never has to guess column order.

### `inference` — 9 nodes

Loads `production_model` plus the four fitted artifacts, applies the identical
transform chain, and writes `data/07_model_output/inference_predictions.json`.

The cleaned inference batch is held to the **same expectation suite as the
training data**, which is what turns a passive pipeline into a drift check.
Expectations naming a column that the batch does not carry are skipped, and
table-level ones are narrowed to the columns present — so a batch with no
`OUTCOME` at all, which is the normal case in production, is validated rather
than rejected for the label it is not supposed to have.

### The API

Scoring calls the same node functions the batch pipeline uses. Two things keep
it fast under load:

- **`bootstrap_project` runs exactly once per process**, behind a double-checked
  lock, and a single `KedroSession` is opened at startup to read the parameters
  and unpickle the artifacts. No request pays for a session, a config load or a
  disk read of the model.
- **POST handlers are `async`** and hand the blocking, CPU-bound scoring chain to
  a worker thread via `run_in_threadpool`, so a slow prediction cannot stall the
  event loop while other requests are parsed and validated.

---

## 12. Results

Held-out numbers, stratified 70/30 split, `random_state=17`:

| model | split | accuracy | precision | recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| LogisticRegression (baseline) | test | 0.719 | 0.597 | 0.623 | 0.610 | **0.817** |
| RandomForest (tuned, 27-point grid) | test | 0.714 | 0.589 | 0.623 | 0.606 | 0.811 |

`select_best_model` promoted the **baseline LogisticRegression** — the tuned
forest wins cross-validation (0.839) but loses on the held-out split, a textbook
case of a grid overfitting CV folds at n=456.

Scored against the 116 never-seen inference rows: **accuracy 0.733, ROC AUC
0.814, F1 0.627.** Test AUC 0.817 vs holdout AUC 0.814 is the evidence the
leakage fixes below worked — the estimate generalises.

> The notebook reports up to 0.80 accuracy. Those numbers are inflated; see next.

---

## 13. What changed from the notebook

The notebook is sound exploratory work, but several steps do not survive contact
with a pipeline that has to score unseen data.

| # | Notebook | Here | Why |
|---|---|---|---|
| 1 | `KNNImputer` and both `RobustScaler`s fit on all 652 rows **before** `train_test_split` | fit on the train split only, persisted as catalog artifacts | Test values fed the neighbour search. `Insulin` is 48% imputed and `SkinThickness` 29%, so roughly half of two features was contaminated. |
| 2 | Outlier fences computed from the full dataset (and applied to `Outcome`) | fit on train rows, features only | Same leak; capping the target is meaningless. |
| 3 | `pd.get_dummies` on the full frame | `OneHotEncoder(handle_unknown="ignore")` fitted and persisted | The dummy *schema* was defined by whatever data was present. Replaying the notebook's chain on the inference CSV yields **24 columns, not 25** — `NEW_AGE_GLUCOSE_NOM_lowsenior` vanishes. A single-row API request would produce far fewer. Unseen levels now encode as all-zeros against a fixed schema — and are logged, because an all-zero group is otherwise indistinguishable from a real row to everything downstream. |
| 4 | `recall_score(y_pred, y_test)` — arguments reversed in every metric call | `(y_true, y_pred)` | The notebook's "Recall" column is actually precision, and vice versa. |
| 5 | `roc_auc_score` fed hard 0/1 labels | fed `predict_proba(...)[:, 1]` | AUC on thresholded labels is not AUC. |
| 6 | Unstratified split | `stratify=y` | 35% positive at n=652; train drifted to 36.8% positive. |
| 7 | Column lists derived from the data (`min() == 0`, cardinality thresholds) | frozen in `parameters.yml` | `Glucose` has no zeros in the inference CSV, so the notebook's own rule would **not** treat its zeros as missing there. Config, not inference. |
| 8 | Grid search over already-globally-scaled data | search runs on the train split of correctly-fitted features | Every `best_score_` in the notebook is optimistic. |
| 9 | `NEW_AGE_BMI_NOM` used `BMI > 18.5` in its last two branches, overwriting the healthy/overweight rows | the intended BMI-band × age-band cross | Only 3 of 8 intended labels ever appeared. All 7 observed combinations now survive. |
| 10 | `NEW_INSULIN_SCORE` | dropped | Its "normal" branch returned `None`, leaving one level, which `drop_first` then removed — it contributed zero columns. |
| 11 | `LabelEncoder` imported and defined | dropped | `binary_cols` evaluated to `[]`; it was never applied to anything. |
| 12 | Nine algorithms, none persisted, none selected | two candidates, evaluated, best promoted to `production_model` | The notebook ends without an artifact to serve. |

Columns `NEW_GLUCOSE * INSULIN` / `NEW_GLUCOSE * PREGNANCIES` were renamed to
`NEW_GLUCOSE_INSULIN` / `NEW_GLUCOSE_PREGNANCIES` — whitespace and `*` break
several gradient-boosting backends.

---

## 14. Configuration reference

Everything tunable lives in `conf/base/parameters.yml`; no code edit is needed.

| key | effect |
|---|---|
| `columns.*` | the feature contract — target, raw numerics, sentinel-zero columns, engineered columns. Add a feature here and encoders, scalers and the model all follow. |
| `split` | `test_size`, `random_state`, `stratify` |
| `imputer.n_neighbors` | KNN imputation neighbours |
| `outliers` | `q_low`, `q_high`, `factor` for the Tukey fences |
| `feature_engineering` | age threshold, BMI and glucose bin edges and labels |
| `baseline` / `tuning` | `class_path`, `init_args`, `param_grid`, `cv`, `scoring` |
| `selection` | which `metric` on which `split` decides promotion |
| `inference.threshold` | decision threshold (default 0.5) |
| `data_quality.cleaned` | the suite guarding both training and inference input |
| `data_quality.master_table` | the pipeline's post-conditions |
| `data_quality.*.fail_on_error` | `true` blocks the run, `false` reports and continues |

Change a model without touching Python:

```yaml
tuning:
  class_path: sklearn.ensemble.HistGradientBoostingClassifier
  init_args: {random_state: 46}
  param_grid:
    max_iter: [100, 200]
    learning_rate: [0.05, 0.1]
```

Datasets and their file paths live in `conf/base/catalog.yml`. Put credentials
or machine-specific overrides in `conf/local/` — it is gitignored.

---

## 15. Project layout

```
.github/
  workflows/ci.yml    format + lint + tests, pipeline end to end, Docker smoke test
  pull_request_template.md
conf/base/
  catalog.yml         every dataset, tagged with its kedro-viz layer
  parameters.yml      column contract, split, model class_path + grid
conf/local/           gitignored local overrides
data/
  01_raw/             both source CSVs (committed)
  02_intermediate/ … 08_reporting/    generated by kedro run
src/diabetes/
  api.py              FastAPI service
  validation.py       Great Expectations suites, config-driven
  settings.py         Kedro config-loader settings
  pipeline_registry.py
  pipelines/
    data_engineering/{nodes,pipeline}.py
    modelling/{nodes,pipeline}.py
    inference/{nodes,pipeline}.py
scripts/
  data_quality_demo.py   what each validation layer catches (make quality)
notebooks/            the original exploratory notebook, unchanged
tests/                56 tests
Dockerfile            trains during build; serves uvicorn
docker-compose.yml    one api service on :8000
Makefile              make help
```

Data layers follow the Kedro convention: `01_raw → 02_intermediate → 03_primary
→ 04_feature → 05_model_input → 06_models → 07_model_output → 08_reporting`.
Generated files are gitignored; `data/01_raw/` is committed so the repo clones
and runs.
