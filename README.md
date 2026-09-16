# Diabetes Prediction — Kedro

Production port of `diabetes-prediction.ipynb`: three Kedro pipelines
(data engineering → modelling → inference), a FastAPI service, and a
self-contained Docker image.

Dataset: 768 Pima records, split into 652 modelling rows and 116 held-out
inference rows. Target `Outcome` is imbalanced (~35% positive).

---

## Quickstart

```bash
uv sync --extra dev     # create .venv and install (uv.lock is committed)
uv run kedro run        # data engineering + modelling + inference, ~5s
uv run kedro viz        # open the pipeline graph at localhost:4141
```

Run one pipeline at a time:

```bash
uv run kedro run --pipeline data_engineering
uv run kedro run --pipeline modelling
uv run kedro run --pipeline inference
uv run kedro run --pipeline training     # data_engineering + modelling
```

`make help` lists the same commands as Make targets.

---

## Pipelines

### `data_engineering` — 11 nodes, raw CSV → master table

```
raw_modelling_data
  → clean_data              upper-case schema, sentinel 0 → NaN
  → split_data              stratified train/test tag in a SPLIT column
  → fit_imputer  ─────────┐ RobustScaler + KNNImputer(k=5)   [fit: train only]
  → apply_imputer ────────┘
  → fit_outlier_bounds ───┐ Tukey fences on p05/p95          [fit: train only]
  → apply_outlier_bounds ─┘
  → engineer_features       7 derived columns (stateless)
  → fit_encoder ──────────┐ OneHotEncoder(handle_unknown=ignore) [fit: train only]
  → apply_encoder ────────┘
  → fit_scaler ───────────┐ RobustScaler on 10 numeric columns   [fit: train only]
  → apply_scaler ─────────┘
  → master_table            652 × 35
```

Every stateful step is a **`fit_*` / `apply_*` pair**. The `fit_*` node emits an
artifact the catalog persists (`data/06_models/`); the `apply_*` node consumes it
and never refits. That is the whole design:

- `fit_*` nodes read only rows where `SPLIT == "train"`, so nothing from the test
  split can reach a fitted object.
- The inference pipeline **imports the same `apply_*` functions** from
  `data_engineering`, so there is exactly one implementation of clean / impute /
  cap / engineer / encode / scale. Training-serving skew is structurally
  impossible rather than merely unlikely.

### `modelling` — 5 nodes

Trains a baseline and a grid-searched challenger, evaluates both on the held-out
split, and promotes the winner to `production_model`.

The estimator is a `class_path` string in `conf/base/parameters.yml`, so swapping
`RandomForestClassifier` for `HistGradientBoostingClassifier` or
`lightgbm.LGBMClassifier` is a one-line config change with no code edit.

Each model artifact is self-describing — `{estimator, target_column,
feature_columns, eval_splits}` — so inference never has to guess column order.

### `inference` — 8 nodes

Loads `production_model` plus the four fitted preprocessing artifacts, applies
the identical transform chain to `diabetes-dataset-inference.csv`, and writes
`data/07_model_output/inference_predictions.json`.

---

## Results

Honest held-out numbers, `random_state=17`, stratified 70/30 split:

| model | split | accuracy | precision | recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| LogisticRegression (baseline) | test | 0.719 | 0.597 | 0.623 | 0.610 | **0.817** |
| RandomForest (tuned, 27-point grid) | test | 0.714 | 0.589 | 0.623 | 0.606 | 0.811 |

`select_best_model` promoted the **baseline LogisticRegression** — the tuned
forest wins cross-validation (0.839) but loses on the held-out split, a textbook
case of the grid overfitting CV folds on n=456.

Scored against the 116 never-seen inference rows: **accuracy 0.733, ROC AUC
0.814, F1 0.627.** Test AUC 0.817 vs holdout AUC 0.814 is the signal that the
leakage fixes below worked — the estimate generalises.

> The notebook reports higher accuracy (up to 0.80 for LightGBM). Those numbers
> are inflated: see below.

---

## What changed from the notebook, and why

The notebook is sound exploratory work, but several steps do not survive contact
with a pipeline that has to score unseen data.

| # | Notebook | Here | Why |
|---|---|---|---|
| 1 | `KNNImputer` and both `RobustScaler`s fit on all 652 rows **before** `train_test_split` | fit on the train split only, persisted as catalog artifacts | Test values fed the neighbour search. `Insulin` is 48% imputed and `SkinThickness` 29%, so roughly half of two features was contaminated. |
| 2 | Outlier fences computed from the full dataset (and applied to `Outcome`) | fit on train rows, features only | Same leak; capping the target is meaningless. |
| 3 | `pd.get_dummies` on the full frame | `OneHotEncoder(handle_unknown="ignore")` fitted and persisted | The dummy *schema* was defined by whatever data was present. Re-running the notebook's chain on the inference CSV yields **24 columns, not 25** — `NEW_AGE_GLUCOSE_NOM_lowsenior` vanishes. A single-row API request would produce far fewer. Now unseen levels encode as all-zeros against a fixed schema. |
| 4 | `recall_score(y_pred, y_test)` — arguments reversed in every metric call | `(y_true, y_pred)` | The notebook's "Recall" column is actually precision and vice versa. |
| 5 | `roc_auc_score` fed hard 0/1 labels | fed `predict_proba(...)[:, 1]` | AUC on thresholded labels is not AUC. |
| 6 | Unstratified split | `stratify=y` | 35% positive on n=652; train drifted to 36.8% positive. |
| 7 | Column lists derived from the data (`min() == 0`, cardinality thresholds) | frozen in `parameters.yml` | `Glucose` has no zeros in the inference CSV, so the notebook's own rule would **not** treat its zeros as missing there. Config, not inference. |
| 8 | Grid search over already-globally-scaled data | search runs on the train split of correctly-fitted features | Every `best_score_` in the notebook is optimistic. |
| 9 | `NEW_AGE_BMI_NOM` used `BMI > 18.5` in its last two branches, overwriting the healthy/overweight rows | the intended BMI-band × age-band cross | Only 3 of 8 intended labels ever appeared. All 7 observed combinations now survive. |
| 10 | `NEW_INSULIN_SCORE` | dropped | Its "normal" branch returned `None`, leaving one level, which `drop_first` then removed — it contributed zero columns. |
| 11 | `LabelEncoder` imported and defined | dropped | `binary_cols` evaluated to `[]`; it was never applied to anything. |
| 12 | Nine algorithms, none persisted, none selected | two candidates, evaluated, best one promoted to `production_model` | The notebook ends without an artifact to serve. |

Column names `NEW_GLUCOSE * INSULIN` / `NEW_GLUCOSE * PREGNANCIES` were renamed
to `NEW_GLUCOSE_INSULIN` / `NEW_GLUCOSE_PREGNANCIES` — whitespace and `*` break
several gradient-boosting backends.

---

## API

```bash
uv run uvicorn diabetes.api:app --reload    # then open localhost:8000/docs
```

Scoring calls the same node functions the batch pipeline uses, so the HTTP path
and the `kedro run --pipeline inference` path cannot diverge.

Two things keep it fast under load:

- **`bootstrap_project` runs exactly once per process**, behind a double-checked
  lock, and a single `KedroSession` is opened at startup to read the parameters
  and unpickle the artifacts. No request pays for a session, a config load or a
  disk read of the model. If the API starts before the first `kedro run`,
  `/ready` returns 503 and retries the load — no restart needed.
- **POST handlers are `async`** and hand the blocking, CPU-bound scoring chain to
  a worker thread via `run_in_threadpool`, so a slow prediction cannot stall the
  event loop while other requests are being parsed and validated.

| method | route | purpose |
|---|---|---|
| GET | `/health` | liveness — the process is up |
| GET | `/ready` | readiness — artifacts loaded; 503 before the first `kedro run` |
| POST | `/predict` | score one patient |
| POST | `/predict/batch` | score up to 1000 patients |
| GET | `/datasets` | list the catalog datasets exposed over HTTP |
| GET | `/datasets/{name}` | read a dataset as JSON, `?limit=&offset=` |

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{
  "Pregnancies": 6, "Glucose": 183, "BloodPressure": 72, "SkinThickness": 35,
  "Insulin": 0, "BMI": 38.6, "DiabetesPedigreeFunction": 1.2, "Age": 55}'
# {"row":0,"probability":0.968447,"prediction":1}

curl 'localhost:8000/datasets/inference_predictions?limit=2'
```

The request model takes the **8 raw measurements** and runs the full transform
chain server-side — clients never see the 33 engineered features. `extra="forbid"`
means a typo'd field is a 422, not a silent wrong answer. `0` stays legal on the
five sentinel columns because that is how "not measured" is encoded.

---

## Docker

```bash
docker compose up --build       # → localhost:8000/docs
```

The image trains during the build (`kedro run --pipeline training`), so it is
self-contained: no model pickles in git, no volumes to mount, and the artifacts
provably match the code in the image. Retraining means rebuilding.

---

## Layout

```
conf/base/
  catalog.yml         every dataset, tagged with its kedro-viz layer
  parameters.yml      column contract, split, model class_path + grid
data/
  01_raw/             both source CSVs (committed)
  02_intermediate/ … 08_reporting/    generated by kedro run
src/diabetes/
  api.py              FastAPI service
  pipeline_registry.py
  pipelines/{data_engineering,modelling,inference}/
notebooks/            the original exploratory notebook, unchanged
tests/                24 tests: uv run pytest
```

Data layers follow the Kedro convention: `01_raw → 02_intermediate → 03_primary
→ 04_feature → 05_model_input → 06_models → 07_model_output → 08_reporting`.
Generated files are gitignored; `data/01_raw/` is committed so the repo clones
and runs.

---

## Tests

```bash
uv run pytest -q     # 24 passed
uv run ruff check src tests
```

The suite targets the contracts that matter rather than line coverage: fitted
artifacts never see test rows, the encoder emits an identical schema for a
single row and for an unseen category, the model artifact never treats `SPLIT`
or the target as a feature, the API rejects malformed payloads, and serving
never re-bootstraps Kedro or reopens a session.

Note `tests/test_api.py` expects `kedro run` to have produced artifacts first.
