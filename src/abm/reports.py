"""Markdown report generation."""

from __future__ import annotations

import pandas as pd


def flatten_aggregate_metrics(metrics: list[dict]) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()
    return pd.json_normalize(metrics)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional pandas deps."""
    if frame.empty:
        return ""
    text_frame = frame.astype(str)
    columns = list(text_frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in text_frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def generate_markdown_report(
    *,
    project_name: str,
    model_name: str,
    dataset_name: str,
    category: str | None,
    checkpoint: str,
    aggregate_metrics: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    scalar_metrics: dict,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    defect_breakdown: pd.DataFrame,
) -> str:
    cm = confusion_matrix.iloc[0].to_dict() if not confusion_matrix.empty else {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    threshold = threshold_summary.iloc[0].to_dict() if not threshold_summary.empty else {}
    lines = [
        f"# {project_name}",
        "",
        "## Run",
        "",
        f"- Model: `{model_name}`",
        f"- Dataset: `{dataset_name}`",
        f"- Category: `{category}`",
        f"- Checkpoint: `{checkpoint}`",
        "",
        "## Aggregate Anomalib Metrics",
        "",
    ]
    if aggregate_metrics.empty:
        lines.append("No aggregate metrics were returned.")
    else:
        lines.append(markdown_table(aggregate_metrics))
    lines.extend(
        [
            "",
            "## Threshold",
            "",
            f"- Method: `{threshold.get('method', 'unknown')}`",
            f"- Threshold: `{threshold.get('threshold', 'unknown')}`",
            "",
            "## Confusion Matrix",
            "",
            f"- TN: {cm.get('tn', 0)}",
            f"- FP: {cm.get('fp', 0)}",
            f"- FN: {cm.get('fn', 0)}",
            f"- TP: {cm.get('tp', 0)}",
            "",
            "## Report Metrics",
            "",
            f"- Images: {scalar_metrics.get('num_images', 0)}",
            f"- Accuracy: {scalar_metrics.get('accuracy', 0.0):.4f}",
            f"- Precision: {scalar_metrics.get('precision', 0.0):.4f}",
            f"- Recall: {scalar_metrics.get('recall', 0.0):.4f}",
            f"- F1: {scalar_metrics.get('f1', 0.0):.4f}",
            "",
            "## Inspection Counts",
            "",
            f"- Good images rejected: {len(false_positives)}",
            f"- Bad images passed as good: {len(false_negatives)}",
            "",
            "## Top Failing Defect Types",
            "",
        ]
    )
    if defect_breakdown.empty:
        lines.append("No defect-type failures were available.")
    else:
        lines.append(markdown_table(defect_breakdown.head(10)))
    lines.extend(
        [
            "",
            "## Practical Interpretation",
            "",
            "Review false negatives first when missed defects are more costly than rejecting good images. "
            "Review false positives first when unnecessary rejection or manual inspection load is the main issue. "
            "Use the threshold sweep as report-level analysis only; Anomalib remains the source of model predictions.",
            "",
        ]
    )
    return "\n".join(lines)
