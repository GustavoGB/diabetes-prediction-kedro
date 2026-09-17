.PHONY: help install train infer versions rollback diagram quality viz api test format lint docker-build docker-up docker-down
.DEFAULT_GOAL := help

install:  ## Create the venv and install everything (incl. dev extras)
	uv sync --extra dev

train:  ## Run the data engineering + modelling pipelines
	uv run kedro run --pipeline training

infer:  ## Score data/01_raw/diabetes-dataset-inference.csv
	uv run kedro run --pipeline inference

versions:  ## List the saved artifact versions, newest last
	@ls -1 data/06_models/production_model.pkl 2>/dev/null || \
	  echo "no versions yet — run 'make train'"

rollback:  ## Score with an earlier artifact set: make rollback VERSION=<timestamp>
	@test -n "$$VERSION" || { echo "usage: make rollback VERSION=<timestamp>  (see: make versions)"; exit 1; }
	uv run kedro run --pipeline inference \
	  $(foreach a,imputer outlier_bounds encoder scaler production_model,--load-versions=$(a):$$VERSION)

quality:  ## Show what each validation layer catches
	uv run python scripts/data_quality_demo.py

diagram:  ## Regenerate docs/pipelines.md from the pipeline definitions
	uv run python scripts/export_pipeline_diagram.py

viz:  ## Open the Kedro pipeline visualisation
	uv run kedro viz

api:  ## Serve the FastAPI app on :8000
	uv run uvicorn diabetes.api:app --reload --port 8000

test:  ## Run the test suite
	uv run pytest -q

format:  ## Format with black
	uv run black src tests scripts

lint:  ## Check formatting and lint
	uv run black --check src tests scripts
	uv run ruff check src tests scripts

docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Start the API container on :8000
	docker compose up -d

docker-down:  ## Stop the API container
	docker compose down

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
