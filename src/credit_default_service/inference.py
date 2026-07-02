"""Model loading, request validation, and prediction helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_default_service.config import (
    DEFAULT_THRESHOLD,
    FEATURE_RULES,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_FILENAMES,
)


@dataclass(frozen=True)
class PredictionResult:
    """A normalized response from a model prediction."""

    model_version: str
    ab_group: str
    prediction: int
    default_probability: float
    threshold: float
    risk_level: str


class ModelRegistry:
    """Loads saved model artifacts and chooses a model for inference."""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self._artifacts: dict[str, dict[str, Any]] = {}

    @property
    def available_versions(self) -> list[str]:
        return sorted(self._artifacts)

    def load(self) -> None:
        """Load all configured model versions from disk."""

        artifacts: dict[str, dict[str, Any]] = {}
        for version, filename in MODEL_FILENAMES.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                artifact = joblib.load(model_path)
                artifacts[version] = artifact

        if not artifacts:
            expected = ", ".join(MODEL_FILENAMES.values())
            raise FileNotFoundError(
                f"No model artifacts found in {self.model_dir}. Expected: {expected}"
            )

        self._artifacts = artifacts

    def choose_version(self, payload: dict[str, Any], requested_version: str | None) -> tuple[str, str]:
        """Choose explicit model version or stable 50/50 A/B assignment."""

        if requested_version:
            if requested_version not in self._artifacts:
                raise ValueError(
                    f"Unknown model_version={requested_version!r}. "
                    f"Available versions: {self.available_versions}"
                )
            return requested_version, "control" if requested_version == "v1" else "test"

        if len(self._artifacts) == 1 or "v2" not in self._artifacts:
            version = self.available_versions[0]
            return version, "control"

        stable_key = str(payload.get("ID") or json.dumps(payload, sort_keys=True))
        bucket = int(hashlib.sha256(stable_key.encode("utf-8")).hexdigest(), 16) % 100
        return ("v1", "control") if bucket < 50 else ("v2", "test")

    def predict(self, payload: dict[str, Any], requested_version: str | None = None) -> PredictionResult:
        version, ab_group = self.choose_version(payload, requested_version)
        artifact = self._artifacts[version]
        features = normalize_features(payload, artifact.get("feature_columns", FEATURE_COLUMNS))
        probabilities = np.asarray(artifact["pipeline"].predict_proba(features))
        probability = float(probabilities[0, 1])
        threshold = float(artifact.get("threshold", DEFAULT_THRESHOLD))
        prediction = int(probability >= threshold)
        return PredictionResult(
            model_version=version,
            ab_group=ab_group,
            prediction=prediction,
            default_probability=round(probability, 6),
            threshold=threshold,
            risk_level=get_risk_level(probability),
        )


def normalize_payload(raw_payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Accept either a flat JSON object or {"features": {...}} envelope."""

    requested_version = raw_payload.get("model_version")
    features = raw_payload.get("features", raw_payload)
    if not isinstance(features, dict):
        raise ValueError("Field 'features' must be a JSON object when provided.")
    return features, requested_version


def normalize_features(payload: dict[str, Any], feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Validate and coerce model features into a one-row DataFrame."""

    columns = feature_columns or FEATURE_COLUMNS
    missing = [column for column in columns if column not in payload]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    row: dict[str, Any] = {}
    for column in columns:
        value = payload[column]
        if value is None:
            raise ValueError(f"Feature {column!r} cannot be null.")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature {column!r} must be numeric.") from exc

        rule = FEATURE_RULES.get(column, {})
        if "allowed" in rule:
            if not float(numeric_value).is_integer() or int(numeric_value) not in rule["allowed"]:
                raise ValueError(f"Feature {column!r} must be one of {rule['allowed']}.")
        if "min" in rule and numeric_value < rule["min"]:
            raise ValueError(f"Feature {column!r} must be >= {rule['min']}.")
        if "max" in rule and numeric_value > rule["max"]:
            raise ValueError(f"Feature {column!r} must be <= {rule['max']}.")

        row[column] = int(numeric_value) if float(numeric_value).is_integer() else numeric_value

    return pd.DataFrame([row], columns=columns)


def get_risk_level(default_probability: float) -> str:
    if default_probability < 0.3:
        return "low"
    if default_probability < 0.6:
        return "medium"
    return "high"
