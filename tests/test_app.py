from pathlib import Path

import joblib

from credit_default_service.app import create_app


class DummyPipeline:
    def predict_proba(self, features):
        return [[0.8, 0.2]]


def test_health_and_predict_with_dummy_model(tmp_path):
    artifact = {
        "version": "v1",
        "pipeline": DummyPipeline(),
        "feature_columns": ["LIMIT_BAL"],
        "threshold": 0.5,
        "metrics": {},
    }
    joblib.dump(artifact, Path(tmp_path) / "credit_default_v1.joblib")

    app = create_app(model_dir=tmp_path)
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["available_models"] == ["v1"]

    response = client.post("/predict", json={"features": {"LIMIT_BAL": 100000}})
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_version"] == "v1"
    assert body["prediction"] == 0
    assert body["default_probability"] == 0.2
    assert body["risk_level"] == "low"

    batch_response = client.post("/batch_predict", json={"items": [{"LIMIT_BAL": 100000}]})
    assert batch_response.status_code == 200
    assert len(batch_response.get_json()["predictions"]) == 1
