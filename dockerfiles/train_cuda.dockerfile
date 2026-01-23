# Base image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ENV TZ=UTC
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required to build Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl git build-essential unzip \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev libffi-dev liblzma-dev tk-dev uuid-dev && \
    rm -rf /var/lib/apt/lists/*

# Build and install Python 3.13 from source
RUN wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz && \
    tar -xzf Python-3.13.0.tgz && \
    cd Python-3.13.0 && \
    ./configure && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd .. && rm -rf Python-3.13.0 Python-3.13.0.tgz

# Make python3.13 the default python
RUN ln -sf /usr/local/bin/python3.13 /usr/bin/python && \
    ln -sf /usr/local/bin/python3.13 /usr/bin/python3 && \
    python --version

# ---------------------------------------------------------
# Install uv (fast Python package manager)
# ---------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ---------------------------------------------------------
# Set working directory
# ---------------------------------------------------------
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

#Install project code
RUN uv sync --frozen

# Make src importable
ENV PYTHONPATH="/app/src:\${PYTHONPATH}"

# ---------------------------------------------------------
# Default command (can be overridden)
# ---------------------------------------------------------
# Example: run your training script
CMD ["uv", "run", "train"]
