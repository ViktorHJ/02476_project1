FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

WORKDIR /app

# Copy dependency files
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

# Install dependencies (but not project code yet)
RUN uv sync --frozen --no-install-project

# Copy source code and pretrained model
COPY src/ src/
COPY models/ models/
COPY README.md README.md
COPY LICENSE LICENSE

# Install project code
RUN uv sync --frozen

# Make src importable
ENV PYTHONPATH="/app/src:\${PYTHONPATH}"

# Run FastAPI server
ENTRYPOINT ["uv", "run", "uvicorn", "cifakeclassification.api:app", "--host", "0.0.0.0", "--port", "8000"]
