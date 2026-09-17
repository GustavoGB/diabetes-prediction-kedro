# Troubleshooting

Errors you can hit on a fresh checkout, and what each one means.

## Troubleshooting


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

---

[← back to the README](../README.md)
