import pandas as pd
import pytest

from credit_default_service.config import FEATURE_COLUMNS
from credit_default_service.inference import get_risk_level, normalize_features, normalize_payload


def test_normalize_payload_accepts_envelope():
    payload = {"model_version": "v1", "features": {"LIMIT_BAL": 100000}}

    features, version = normalize_payload(payload)

    assert features == {"LIMIT_BAL": 100000}
    assert version == "v1"


def test_normalize_features_requires_all_model_features():
    with pytest.raises(ValueError, match="Missing required feature"):
        normalize_features({"LIMIT_BAL": 100000})


def test_normalize_features_returns_ordered_dataframe():
    payload = {column: 1 for column in FEATURE_COLUMNS}
    payload.update({"AGE": 35, "LIMIT_BAL": 100000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1})
    frame = normalize_features(payload)

    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == FEATURE_COLUMNS
    assert len(frame) == 1


def test_normalize_features_validates_ranges():
    payload = {column: 1 for column in FEATURE_COLUMNS}
    payload["AGE"] = 17

    with pytest.raises(ValueError, match="AGE"):
        normalize_features(payload)


def test_get_risk_level():
    assert get_risk_level(0.1) == "low"
    assert get_risk_level(0.4) == "medium"
    assert get_risk_level(0.8) == "high"
