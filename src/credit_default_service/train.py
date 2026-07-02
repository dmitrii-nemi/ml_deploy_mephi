"""Train and save credit default model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_default_service.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS,
    ID_COLUMN,
    MODEL_FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_FILENAMES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from credit_default_service.features import add_engineered_features


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(version: str) -> Pipeline:
    if version == "v1":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif version == "v2":
        classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model version: {version}")

    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def evaluate_pipeline(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def train_models(data_path: Path, model_dir: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(data_path)
    x = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_version: dict[str, dict[str, float]] = {}

    for version, filename in MODEL_FILENAMES.items():
        pipeline = build_pipeline(version)
        pipeline.fit(x_train, y_train)
        metrics = evaluate_pipeline(pipeline, x_test, y_test)
        metrics_by_version[version] = metrics

        artifact = {
            "version": version,
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "model_feature_columns": MODEL_FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "threshold": DEFAULT_THRESHOLD,
            "metrics": metrics,
            "training_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
        }
        joblib.dump(artifact, model_dir / filename)

    (model_dir / "metrics.json").write_text(
        json.dumps(metrics_by_version, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sample = df.iloc[0][[ID_COLUMN] + FEATURE_COLUMNS].to_dict()
    (model_dir / "sample_request.json").write_text(
        json.dumps({"model_version": "v1", "features": sample}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return metrics_by_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit default models.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_models(args.data_path, args.model_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
