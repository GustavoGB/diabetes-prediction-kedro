.PHONY: help install train infer viz api test format lint docker-build docker-up docker-down
.DEFAULT_GOAL := help

install:  ## Create the venv and install everything (incl. dev extras)
	uv sync --extra dev

train:  ## Run the data engineering + modelling pipelines
	uv run kedro run --pipeline training

infer:  ## Score data/01_raw/diabetes-dataset-inference.csv
	uv run kedro run --pipeline inference

viz:  ## Open the Kedro pipeline visualisation
	uv run kedro viz

api:  ## Serve the FastAPI app on :8000
	uv run uvicorn diabetes.api:app --reload --port 8000

test:  ## Run the test suite
	uv run pytest -q

format:  ## Format with black
	uv run black src tests

lint:  ## Check formatting and lint
	uv run black --check src tests
	uv run ruff check src tests

docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Start the API container on :8000
	docker compose up -d

docker-down:  ## Stop the API container
	docker compose down

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
