#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/data_processing"

echo "▶ Running full data crawl & database seed..."

python get_users.py
python get_ratings.py
python get_movies.py

echo "▶ Optional maintenance..."
python prune_inactive_movies.py || true
python consolidate_redirects.py || true

echo "✅ Database seed complete."
