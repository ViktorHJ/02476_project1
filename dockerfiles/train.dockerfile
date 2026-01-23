# Default: CPU-friendly base image
ARG BASE_IMAGE=ghcr.io/astral-sh/uv:python3.13-bookworm
FROM ${BASE_IMAGE} AS base

WORKDIR /app


# Copy dependency files
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md
COPY LICENSE LICENSE

# Install dependencies (but not project code yet)
RUN uv sync --frozen --no-install-project

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY .env .env
COPY reports/ reports/

# Install project code
RUN uv sync --frozen

# Make src importable
ENV PYTHONPATH="/app/src:\${PYTHONPATH}"

# Default entrypoint: training
ENTRYPOINT ["uv", "run"]
