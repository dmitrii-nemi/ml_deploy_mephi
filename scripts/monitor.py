"""Small PSI-based monitoring helper for feature drift checks."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default_service.config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_props = expected_counts / max(len(expected), 1)
    actual_props = actual_counts / max(len(actual), 1)
    expected_props = np.where(expected_props == 0, 0.0001, expected_props)
    actual_props = np.where(actual_props == 0, 0.0001, actual_props)

    return float(np.sum((actual_props - expected_props) * np.log(actual_props / expected_props)))


def drift_status(psi: float) -> str:
    if psi >= 0.2:
        return "high"
    if psi >= 0.1:
        return "moderate"
    return "ok"


def get_api_health(api_url: str) -> dict[str, object]:
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/health", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"status": "unavailable", "error": str(exc)}


def run_monitoring(data_path: Path, api_url: str, sample_size: int) -> dict[str, object]:
    df = pd.read_csv(data_path)
    reference = df.sample(frac=0.7, random_state=42)
    current = df.drop(reference.index).sample(n=min(sample_size, len(df) - len(reference)), random_state=43)

    psi_by_feature = {}
    for feature in FEATURE_COLUMNS:
        psi = calculate_psi(reference[feature].to_numpy(), current[feature].to_numpy())
        psi_by_feature[feature] = {"psi": round(psi, 6), "status": drift_status(psi)}

    return {
        "rows": int(len(df)),
        "target_rate": round(float(df[TARGET_COLUMN].mean()), 6),
        "api_health": get_api_health(api_url),
        "feature_drift": psi_by_feature,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSI feature drift monitoring.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--api-url", default="http://127.0.0.1:5000")
    parser.add_argument("--sample-size", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(run_monitoring(args.data_path, args.api_url, args.sample_size), indent=2))


if __name__ == "__main__":
    main()
