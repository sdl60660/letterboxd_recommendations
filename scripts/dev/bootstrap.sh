# scripts/dev/bootstrap.sh
#!/usr/bin/env bash
set -euo pipefail

echo "▶ Installing Python tooling…"
python -m pip install --upgrade pip pipenv

echo "▶ Installing dev deps (incl. pre-commit)…"
pipenv install --dev

echo "▶ Installing pre-commit hooks…"
pipenv run pre-commit install

echo "▶ Frontend deps…"
if command -v npm >/dev/null 2>&1; then
  ( cd frontend && npm ci || npm install )
else
  echo "⚠️  npm not found. Skipping frontend install."
fi

echo "▶ Building & starting Docker services…"
docker compose up -d --build

echo "▶ Waiting for Mongo…"
scripts/dev/wait-for.sh mongo:27017 -t 60

echo "▶ Creating DB indexes (safe/idempotent)…"
docker compose exec -T mongo mongosh "$MONGO_DB" scripts/db/create_indexes.js

echo "▶ Seeding minimal data…"
docker compose exec -T web python scripts/db/seed_minimal.py || true

echo "▶ Running quick tests (skip live)…"
pipenv run pytest -q -m "not live" || (echo "❌ tests failed" && exit 1)

echo "✅ Bootstrap complete. Try:"
echo "   - Backend: http://localhost:8000"
echo "   - Frontend: http://localhost:3000"
