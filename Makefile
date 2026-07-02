.PHONY: help install train api test docker-build docker-run docker-push compose-up compose-down monitor clean

IMAGE ?= batonhleba/mephi_ml:latest
PORT ?= 5000

help:
	@echo "Available commands:"
	@echo "  make install       Install Python dependencies and package"
	@echo "  make train         Train and save model artifacts"
	@echo "  make api           Run Flask API locally"
	@echo "  make test          Run unit tests"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker image locally"
	@echo "  make docker-push   Push Docker image to Docker Hub"
	@echo "  make compose-up    Run with Docker Compose"
	@echo "  make monitor       Run PSI monitoring helper"

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python -m credit_default_service.train

api:
	PORT=$(PORT) python -m credit_default_service

test:
	pytest

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p $(PORT):5000 $(IMAGE)

docker-push:
	docker push $(IMAGE)

compose-up:
	docker compose up --build

compose-down:
	docker compose down

monitor:
	python scripts/monitor.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage
