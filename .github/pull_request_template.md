## What changed

<!-- One or two sentences. Link an issue if there is one. -->

## Checks

- [ ] `uv run pytest -q` passes
- [ ] `uv run black --check src tests scripts` and `uv run ruff check src tests scripts` are clean
- [ ] `uv run kedro run` completes from a clean checkout
- [ ] The three `data/08_reporting/*_quality.json` suites report `"success": true`
- [ ] `docker compose build && docker compose up -d`, then `/ready` returns 200 (only if the image, deps or API changed)
- [ ] README / parameters updated if behaviour or configuration changed
