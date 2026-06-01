"""Report-level threshold analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from .config import ThresholdConfig
from .utils import BenchmarkError


@dataclass
class ThresholdResult:
    per_image: pd.DataFrame
    sweep: pd.DataFrame
    summary: pd.DataFrame
    threshold: float | str
    method: str


def generate_threshold_sweep(df: pd.DataFrame, num_thresholds: int) -> pd.DataFrame:
    valid = df.dropna(subset=["score", "y_true"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])
    y_true = valid["y_true"].astype(int).to_numpy()
    scores = valid["score"].astype(float).to_numpy()
    low, high = float(np.nanmin(scores)), float(np.nanmax(scores))
    if low == high:
        thresholds = np.array([low])
    else:
        thresholds = np.linspace(low, high, num_thresholds)
    rows = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def _best_f1_threshold(sweep: pd.DataFrame) -> float:
    if sweep.empty:
        raise BenchmarkError("Cannot select best_f1 threshold because no valid score/y_true rows are available.")
    best = sweep.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
    return float(best["threshold"])


def apply_threshold(df: pd.DataFrame, config: ThresholdConfig) -> ThresholdResult:
    """Apply report-level threshold analysis to parsed predictions."""
    output = df.copy()
    sweep = generate_threshold_sweep(output, config.num_thresholds)
    if config.method == "best_f1":
        threshold: float | str = _best_f1_threshold(sweep)
        output["threshold_y_pred"] = (output["score"].astype(float) >= float(threshold)).astype(int)
    elif config.method == "manual":
        if config.manual_value is None:
            raise BenchmarkError("manual threshold requires manual_value.")
        threshold = float(config.manual_value)
        output["threshold_y_pred"] = (output["score"].astype(float) >= threshold).astype(int)
    elif config.method == "use_model_threshold":
        threshold = "model_provided"
        if "y_pred" not in output.columns:
            raise BenchmarkError("use_model_threshold requires Anomalib prediction labels in y_pred.")
        output["threshold_y_pred"] = output["y_pred"]
    else:  # pragma: no cover - pydantic validates this
        raise BenchmarkError(f"Unsupported threshold method: {config.method}")

    summary = pd.DataFrame(
        [
            {
                "method": config.method,
                "threshold": threshold,
                "num_thresholds": config.num_thresholds,
                "valid_score_rows": int(output["score"].notna().sum()),
            }
        ]
    )
    return ThresholdResult(output, sweep, summary, threshold, config.method)
