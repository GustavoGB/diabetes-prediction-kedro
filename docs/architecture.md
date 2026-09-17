# How it is built

The pipelines node by node, what changed from the notebook and why, the configuration reference, and the repository layout. For the DAGs themselves see [pipelines.md](pipelines.md).

## How it is built


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
## What changed from the notebook


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
## Configuration reference


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
## Project layout


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
  data_quality_demo.py        what each validation layer catches (make quality)
  export_pipeline_diagram.py  regenerates docs/pipelines.md (make diagram)
docs/                  the long-form documentation the README links to
notebooks/            the original exploratory notebook, unchanged
tests/                59 tests
Dockerfile            trains during build; serves uvicorn
docker-compose.yml    one api service on :8000
Makefile              make help
```

Data layers follow the Kedro convention: `01_raw → 02_intermediate → 03_primary
→ 04_feature → 05_model_input → 06_models → 07_model_output → 08_reporting`.
Generated files are gitignored; `data/01_raw/` is committed so the repo clones
and runs.

---

[← back to the README](../README.md)
