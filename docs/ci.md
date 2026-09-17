# Continuous integration

What runs on every push and pull request, what each job actually asserts, and how to run the same checks locally.

## Continuous integration


Every push to `main` and every pull request targeting `main` runs
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) on GitHub Actions. Three
jobs run in parallel, on a clean `ubuntu-latest` checkout with no cached
artifacts — so CI proves the repository is reproducible from nothing but what is
committed.

| job | what it does | why it is separate |
|---|---|---|
| **Format, lint, tests** | `black --check`, `ruff check`, `kedro run`, then `pytest -q` (59 tests) | fastest feedback; a formatting slip should not wait on Docker |
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
  [data-quality.md](data-quality.md));
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
base image in the [Dockerfile](../Dockerfile), so the tested interpreter is the
shipped one. The lockfile covers `>=3.10,<3.14`, so developing on 3.11 locally is
fine.

### Pull requests

[`.github/pull_request_template.md`](../.github/pull_request_template.md) pre-fills
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

---

[← back to the README](../README.md)
