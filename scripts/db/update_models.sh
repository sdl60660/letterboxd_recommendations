#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/data_processing"

echo "▶ Updating sample data..."
# pipenv run python create_training_data.py || true

echo "▶ Tuning model weights..."
pipenv run python evaluate_models.py || true

echo "▶ Building new models..."
pipenv run python build_model.py || true

echo "✅ Model update complete."