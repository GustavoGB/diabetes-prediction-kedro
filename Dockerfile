FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Dependency layer: cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project layer. README.md is required here: pyproject declares
# readme = "README.md", so the wheel cannot be built without it.
COPY README.md ./
COPY src/ src/
COPY conf/ conf/
COPY data/01_raw/ data/01_raw/
RUN uv sync --frozen --no-dev

# Train inside the image so the container is self-contained and its artifacts
# provably match this code. No pickles in git, nothing to mount at runtime.
RUN uv run kedro run --pipeline training

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["uv", "run", "uvicorn", "diabetes.api:app", "--host", "0.0.0.0", "--port", "8000"]
