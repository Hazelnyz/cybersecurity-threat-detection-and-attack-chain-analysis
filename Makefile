.PHONY: install lint test up down

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest

up:
	docker compose up -d

down:
	docker compose down

