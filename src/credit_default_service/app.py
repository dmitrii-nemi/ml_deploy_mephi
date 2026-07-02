"""Flask application for credit default prediction."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from credit_default_service.config import FEATURE_COLUMNS, LOG_PATH, MODEL_DIR
from credit_default_service.inference import ModelRegistry, normalize_payload


def configure_json_logger(log_path: Path = LOG_PATH) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("credit_default_api")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def create_app(model_dir: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    registry = ModelRegistry(Path(model_dir or os.getenv("MODEL_DIR", MODEL_DIR)))
    logger = configure_json_logger(Path(os.getenv("API_LOG_PATH", LOG_PATH)))

    try:
        registry.load()
        load_error = None
    except Exception as exc:  # pragma: no cover - readiness is exposed through /health
        load_error = str(exc)

    @app.get("/health")
    def health() -> tuple[Any, int]:
        status_code = 200 if load_error is None else 503
        return (
            jsonify(
                {
                    "status": "ok" if load_error is None else "degraded",
                    "service": "credit-default-predictor",
                    "available_models": registry.available_versions,
                    "model_load_error": load_error,
                }
            ),
            status_code,
        )

    @app.get("/")
    def root() -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "service": "credit-default-predictor",
                    "version": "0.1.0",
                    "endpoints": {
                        "health": "/health",
                        "predict": "/predict",
                        "batch_predict": "/batch_predict",
                    },
                }
            ),
            200,
        )

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        if load_error is not None:
            return jsonify({"error": "Model registry is not ready.", "details": load_error}), 503

        raw_payload = request.get_json(silent=True)
        if not isinstance(raw_payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        try:
            features, requested_version = normalize_payload(raw_payload)
            result = registry.predict(features, requested_version)
        except ValueError as exc:
            return jsonify({"error": str(exc), "required_features": FEATURE_COLUMNS}), 400

        response = {
            "model_version": result.model_version,
            "ab_group": result.ab_group,
            "prediction": result.prediction,
            "default_probability": result.default_probability,
            "threshold": result.threshold,
            "risk_level": result.risk_level,
        }
        log_event(logger, features, response)
        return jsonify(response), 200

    @app.post("/batch_predict")
    def batch_predict() -> tuple[Any, int]:
        if load_error is not None:
            return jsonify({"error": "Model registry is not ready.", "details": load_error}), 503

        raw_payload = request.get_json(silent=True)
        if isinstance(raw_payload, dict):
            items = raw_payload.get("items")
            requested_version = raw_payload.get("model_version")
        else:
            items = raw_payload
            requested_version = None

        if not isinstance(items, list) or not items:
            return jsonify({"error": "Request body must be a non-empty JSON array or {'items': [...]}."}), 400

        responses = []
        try:
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("Each batch item must be a JSON object.")
                features, item_version = normalize_payload(item)
                result = registry.predict(features, item_version or requested_version)
                response = {
                    "model_version": result.model_version,
                    "ab_group": result.ab_group,
                    "prediction": result.prediction,
                    "default_probability": result.default_probability,
                    "threshold": result.threshold,
                    "risk_level": result.risk_level,
                }
                log_event(logger, features, response)
                responses.append(response)
        except ValueError as exc:
            return jsonify({"error": str(exc), "required_features": FEATURE_COLUMNS}), 400

        return jsonify({"predictions": responses}), 200

    return app


def log_event(logger: logging.Logger, features: dict[str, Any], response: dict[str, Any]) -> None:
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "client_id": features.get("ID"),
        "model_version": response["model_version"],
        "ab_group": response["ab_group"],
        "prediction": response["prediction"],
        "default_probability": response["default_probability"],
        "risk_level": response["risk_level"],
    }
    logger.info(json.dumps(event, ensure_ascii=False))


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
