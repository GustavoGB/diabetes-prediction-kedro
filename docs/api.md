# The API

Everything the FastAPI service exposes: operations, single and batch scoring, validation errors, and the read-only dataset endpoints. The README covers the three calls you need to see it work; this is the full surface.

## Run the API


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

---

[← back to the README](../README.md)
