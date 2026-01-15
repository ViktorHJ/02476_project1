#!/usr/bin/env bash
set -e

# Fix permissions
sudo chown -R vscode:vscode .

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install --install-hooks

# Auto-activate virtual environment in new shells
echo "source $(pwd)/.venv/bin/activate" >> ~/.bashrc
