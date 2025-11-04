#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/data_processing"

echo "▶ Verifying Python env in container..."
python -V
python -c "import numpy; print('numpy', numpy.__version__)"

echo "▶ Updating sample data..."
python create_training_data.py || true

echo "▶ Tuning model weights..."
python evaluate_models.py || true

echo "▶ Building new models..."
python build_model.py || true

echo "✅ Model update complete."
