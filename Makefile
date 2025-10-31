.PHONY: fmt lint test up api worker
fmt:    ; black . && isort .
lint:   ; flake8 .
test:   ; pytest -q
up:     ; docker compose up --build
api:    ; uvicorn app.main:app --reload
worker: ; celery -A app.tasks worker --loglevel=info
