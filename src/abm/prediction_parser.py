"""Robust parsing for Anomalib prediction outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import LabelFromPathConfig


FIELD_ALIASES = {
    "image_path": ["image_path", "image_paths", "path", "file_path", "filepath"],
    "mask_path": ["mask_path", "mask_paths"],
    "gt_label": ["gt_label", "gt_labels", "label", "labels", "target"],
    "pred_label": ["pred_label", "pred_labels"],
    "pred_score": ["pred_score", "pred_scores", "score", "scores"],
    "defect_type": ["defect_type", "defect_types", "class_name", "class_names"],
    "image_shape": ["image_shape", "image_shapes", "original_shape", "original_shapes"],
    "pixel_score": ["pixel_score", "pixel_scores"],
}

REQUIRED_COLUMNS = ["model", "dataset", "category", "image_path", "defect_type", "group", "y_true", "y_pred", "score"]
OPTIONAL_COLUMNS = ["mask_path", "image_shape", "pixel_score"]


def to_python(value: Any) -> Any:
    """Convert common tensor/array/path values to Python values."""
    if value is None:
        return None
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return value.item()
            return value.tolist()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def unwrap_singletons(value: Any) -> Any:
    """Unwrap nested one-item sequences, preserving path-like strings."""
    value = to_python(value)
    while isinstance(value, list) and len(value) == 1:
        value = to_python(value[0])
    return value


def is_sequence_value(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def flatten_prediction_items(raw_predictions: Any) -> list[Any]:
    """Flatten only the outer prediction container."""
    if raw_predictions is None:
        return []
    raw_predictions = to_python(raw_predictions)
    if isinstance(raw_predictions, list):
        items: list[Any] = []
        for item in raw_predictions:
            if isinstance(item, list):
                items.extend(item)
            else:
                items.append(item)
        return items
    return [raw_predictions]


def get_field(item: Any, canonical_name: str) -> Any:
    aliases = FIELD_ALIASES[canonical_name]
    for alias in aliases:
        if isinstance(item, Mapping) and alias in item:
            return to_python(item[alias])
        if hasattr(item, alias):
            return to_python(getattr(item, alias))
    return None


def _batch_length(fields: Mapping[str, Any]) -> int:
    image_path = fields.get("image_path")
    if is_sequence_value(image_path):
        return len(image_path)
    lengths = [len(value) for value in fields.values() if is_sequence_value(value)]
    return max(lengths, default=1)


def _value_at(value: Any, index: int, batch_len: int) -> Any:
    value = to_python(value)
    if is_sequence_value(value):
        if len(value) == batch_len:
            return unwrap_singletons(value[index])
        if len(value) == 1:
            return unwrap_singletons(value[0])
    return unwrap_singletons(value)


def prediction_item_to_records(item: Any) -> list[dict[str, Any]]:
    fields = {name: get_field(item, name) for name in FIELD_ALIASES}
    batch_len = _batch_length(fields)
    records = []
    for index in range(batch_len):
        records.append({name: _value_at(value, index, batch_len) for name, value in fields.items()})
    return records


def normalize_label(value: Any) -> int | None:
    value = unwrap_singletons(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return 1 if int(value) != 0 else 0
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return 1 if value != 0.0 else 0
    text = str(value).strip().lower()
    if text in {"0", "false", "normal", "good", "ok", "negative"}:
        return 0
    if text in {"1", "true", "abnormal", "anomaly", "anomalous", "bad", "defect", "positive"}:
        return 1
    return None


def infer_label_from_path(path: Any, label_config: LabelFromPathConfig) -> int | None:
    if not path:
        return None
    parts = [part.lower() for part in Path(str(path).replace("\\", "/")).parts]
    normal_tokens = {token.lower() for token in label_config.normal_tokens}
    abnormal_tokens = {token.lower() for token in label_config.abnormal_tokens}
    if any(part in normal_tokens for part in parts):
        return 0
    if any(part in abnormal_tokens for part in parts):
        return 1
    return None


def infer_defect_type(path: Any, y_true: int | None) -> str:
    if y_true == 0:
        return "good"
    if not path:
        return "unknown"
    parts = list(Path(str(path).replace("\\", "/")).parts)
    if len(parts) >= 2:
        parent = parts[-2]
        if parent:
            return parent
    return "unknown"


def parse_predictions(
    raw_predictions: Any,
    *,
    model_name: str,
    dataset_name: str,
    category: str | None,
    label_from_path: LabelFromPathConfig | None = None,
) -> pd.DataFrame:
    """Convert Anomalib predictions into a stable per-image DataFrame."""
    label_config = label_from_path or LabelFromPathConfig()
    records: list[dict[str, Any]] = []
    for item in flatten_prediction_items(raw_predictions):
        records.extend(prediction_item_to_records(item))

    rows: list[dict[str, Any]] = []
    for record in records:
        image_path = record.get("image_path")
        y_true = normalize_label(record.get("gt_label"))
        if y_true is None:
            y_true = infer_label_from_path(image_path, label_config)
        y_pred = normalize_label(record.get("pred_label"))
        defect_type = record.get("defect_type") or infer_defect_type(image_path, y_true)
        score = unwrap_singletons(record.get("pred_score"))
        rows.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "category": category,
                "image_path": str(image_path) if image_path is not None else None,
                "defect_type": str(defect_type) if defect_type is not None else "unknown",
                "group": "bad" if y_true == 1 else "good" if y_true == 0 else "unknown",
                "y_true": y_true,
                "y_pred": y_pred,
                "score": float(score) if score is not None else np.nan,
                "mask_path": str(record.get("mask_path")) if record.get("mask_path") is not None else None,
                "image_shape": record.get("image_shape"),
                "pixel_score": record.get("pixel_score"),
            }
        )

    df = pd.DataFrame(rows)
    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df[REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
