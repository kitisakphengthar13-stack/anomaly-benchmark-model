"""Metrics and diagnostic tables."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def prediction_column(df: pd.DataFrame) -> str:
    return "threshold_y_pred" if "threshold_y_pred" in df.columns else "y_pred"


def valid_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    pred_col = prediction_column(df)
    return df.dropna(subset=["y_true", pred_col]).copy()


def confusion_matrix_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = valid_prediction_rows(df)
    pred_col = prediction_column(valid)
    if valid.empty:
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = confusion_matrix(valid["y_true"].astype(int), valid[pred_col].astype(int), labels=[0, 1]).ravel()
    return pd.DataFrame([{"tn": tn, "fp": fp, "fn": fn, "tp": tp}])


def scalar_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    valid = valid_prediction_rows(df)
    pred_col = prediction_column(valid)
    if valid.empty:
        return {"num_images": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    y_true = valid["y_true"].astype(int)
    y_pred = valid[pred_col].astype(int)
    return {
        "num_images": int(len(valid)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def false_positive_table(df: pd.DataFrame) -> pd.DataFrame:
    pred_col = prediction_column(df)
    return df[(df["y_true"] == 0) & (df[pred_col] == 1)].sort_values("score", ascending=False)


def false_negative_table(df: pd.DataFrame) -> pd.DataFrame:
    pred_col = prediction_column(df)
    return df[(df["y_true"] == 1) & (df[pred_col] == 0)].sort_values("score", ascending=True)


def defect_type_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    pred_col = prediction_column(df)
    valid = df.dropna(subset=["y_true", pred_col]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["defect_type", "count", "fn", "tp", "failure_rate"])
    bad = valid[valid["y_true"] == 1].copy()
    if bad.empty:
        return pd.DataFrame(columns=["defect_type", "count", "fn", "tp", "failure_rate"])
    grouped = bad.groupby("defect_type", dropna=False)
    rows = []
    for defect_type, group in grouped:
        fn = int(((group[pred_col] == 0)).sum())
        tp = int(((group[pred_col] == 1)).sum())
        count = int(len(group))
        rows.append(
            {
                "defect_type": defect_type,
                "count": count,
                "fn": fn,
                "tp": tp,
                "failure_rate": fn / count if count else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["failure_rate", "count"], ascending=[False, False])
