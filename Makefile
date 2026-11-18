.PHONY: fmt lint test up api
fmt:
	isort app tests
	black app tests
lint:
	flake8 app tests
test:
	pytest -q
up:
	docker compose up --build
api:
	uvicorn app.main:app --no-access-log
