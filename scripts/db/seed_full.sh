#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}/data_processing"

echo "▶ Running full data crawl & database seed..."

pipenv run python get_users.py
pipenv run python get_ratings.py
pipenv run python get_movies.py

echo "▶ Optional maintenance..."
pipenv run python prune_inactive_movies.py || true
pipenv run python consolidate_redirects.py || true

echo "✅ Database seed complete."