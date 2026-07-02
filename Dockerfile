FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    PROJECT_ROOT=/app \
    MODEL_DIR=/app/models \
    API_LOG_PATH=/app/logs/api_requests.jsonl

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "credit_default_service.app:app"]
