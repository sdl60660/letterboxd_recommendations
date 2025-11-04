# Makefile
.PHONY: bootstrap up down logs test test-fast lint fmt seed

bootstrap:
	bash scripts/dev/bootstrap.sh

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

lint:
	pipenv run ruff .

fmt:
	pipenv run ruff format .

test:
	pipenv run pytest -q

test-fast:
	pipenv run pytest -q -m "not live"

seed:
	docker compose exec -T web python scripts/db/seed_minimal.py

seed-full:
	bash scripts/db/seed_full.sh
