"""Matplotlib plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import PlotsConfig
from .metrics import confusion_matrix_table, prediction_column, scalar_metrics


def _figure(config: PlotsConfig):
    return plt.subplots(figsize=config.figure_size, dpi=config.dpi)


def _numeric_threshold(threshold: float | str | None) -> float | None:
    if threshold is None:
        return None
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return None


def _threshold_label(threshold: float | str | None) -> str:
    numeric = _numeric_threshold(threshold)
    if numeric is None:
        return str(threshold) if threshold is not None else "not available"
    return f"{numeric:.3f}"


def _add_threshold_line(ax, threshold: float | str | None, config: PlotsConfig, *, orientation: str = "vertical") -> None:
    numeric = _numeric_threshold(threshold)
    if numeric is None:
        return
    label = f"Threshold = {numeric:.3f}"
    if orientation == "horizontal":
        ax.axhline(numeric, color=config.colors.threshold, linestyle="--", linewidth=2, label=label)
    else:
        ax.axvline(numeric, color=config.colors.threshold, linestyle="--", linewidth=2, label=label)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def plot_confusion_matrix(df: pd.DataFrame, output_path: Path, config: PlotsConfig) -> None:
    table = confusion_matrix_table(df).iloc[0]
    matrix = [[table["tn"], table["fp"]], [table["fn"], table["tp"]]]
    labels = [
        [f"TN\n{table['tn']}\nGood -> Good", f"FP\n{table['fp']}\nGood -> Bad"],
        [f"FN\n{table['fn']}\nBad -> Good", f"TP\n{table['tp']}\nBad -> Bad"],
    ]
    metrics = scalar_metrics(df)
    fig, ax = _figure(config)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred good", "Pred bad"])
    ax.set_yticks([0, 1], labels=["True good", "True bad"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_title(
        "Confusion Matrix\n"
        f"Accuracy: {_format_pct(metrics['accuracy'])} | "
        f"Precision: {_format_pct(metrics['precision'])} | "
        f"Recall: {_format_pct(metrics['recall'])} | "
        f"F1: {_format_pct(metrics['f1'])}"
    )
    if config.confusion_matrix.show_values:
        max_value = max(max(row) for row in matrix) or 1
        for y in range(2):
            for x in range(2):
                text_color = "white" if matrix[y][x] > max_value * 0.5 else "black"
                ax.text(x, y, labels[y][x], ha="center", va="center", color=text_color, fontsize=11, fontweight="bold")
    if config.confusion_matrix.colorbar:
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_score_distribution(df: pd.DataFrame, output_path: Path, config: PlotsConfig, threshold: float | str | None = None) -> None:
    fig, ax = _figure(config)
    good = df[df["y_true"] == 0]["score"].dropna()
    bad = df[df["y_true"] == 1]["score"].dropna()
    ax.hist(
        good,
        bins=config.score_distribution.bins,
        alpha=config.score_distribution.alpha,
        label="Good images",
        color=config.colors.good,
        edgecolor="white",
    )
    ax.hist(
        bad,
        bins=config.score_distribution.bins,
        alpha=config.score_distribution.alpha,
        label="Bad images",
        color=config.colors.bad,
        edgecolor="white",
    )
    _add_threshold_line(ax, threshold, config)
    cm = confusion_matrix_table(df).iloc[0]
    summary = f"Threshold: {_threshold_label(threshold)}\nFP: {cm['fp']}\nFN: {cm['fn']}"
    ax.text(
        0.98,
        0.95,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Image Count")
    ax.set_title("Anomaly Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_per_image_score(df: pd.DataFrame, output_path: Path, config: PlotsConfig, threshold: float | str | None = None) -> None:
    plot_df = df.dropna(subset=["score", "y_true"]).copy()
    if config.per_image_score.sort_by_score:
        plot_df = plot_df.sort_values("score", ascending=True).reset_index(drop=True)
        title = "Sorted Per-image Anomaly Scores"
        xlabel = "Images sorted by anomaly score"
    else:
        plot_df = plot_df.reset_index(drop=True)
        title = "Per-image Anomaly Scores"
        xlabel = "Image Index"
    pred_col = prediction_column(plot_df)
    correct_good = plot_df[(plot_df["y_true"] == 0) & (plot_df[pred_col] == 0)]
    correct_bad = plot_df[(plot_df["y_true"] == 1) & (plot_df[pred_col] == 1)]
    false_positive = plot_df[(plot_df["y_true"] == 0) & (plot_df[pred_col] == 1)]
    false_negative = plot_df[(plot_df["y_true"] == 1) & (plot_df[pred_col] == 0)]

    fig, ax = _figure(config)
    marker_size = config.per_image_score.marker_size
    alpha = config.per_image_score.alpha
    ax.scatter(correct_good.index, correct_good["score"], color=config.colors.good, alpha=alpha, s=marker_size, label="Good - correct")
    ax.scatter(correct_bad.index, correct_bad["score"], color=config.colors.bad, alpha=alpha, s=marker_size, label="Bad - correct")
    ax.scatter(
        false_positive.index,
        false_positive["score"],
        color=config.colors.wrong,
        marker="x",
        s=marker_size * 1.8,
        linewidths=2,
        label="False Positive: good predicted bad",
    )
    ax.scatter(
        false_negative.index,
        false_negative["score"],
        color=config.colors.false_negative,
        marker="x",
        s=marker_size * 1.8,
        linewidths=2,
        label="False Negative: bad predicted good",
    )
    _add_threshold_line(ax, threshold, config, orientation="horizontal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Anomaly Score")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_threshold_sweep(sweep: pd.DataFrame, output_path: Path, config: PlotsConfig, threshold: float | str | None = None) -> None:
    fig, ax = _figure(config)
    best_row = None
    if not sweep.empty:
        ax.plot(sweep["threshold"], sweep["precision"], label="Precision")
        ax.plot(sweep["threshold"], sweep["recall"], label="Recall")
        ax.plot(sweep["threshold"], sweep["f1"], label="F1", color=config.colors.threshold)
        best_row = sweep.sort_values(["f1", "recall", "precision", "threshold"], ascending=[False, False, False, True]).iloc[0]
        ax.scatter([best_row["threshold"]], [best_row["f1"]], color=config.colors.threshold, s=70, zorder=5, label="Best F1")
        ax.annotate(
            "Best F1",
            xy=(best_row["threshold"], best_row["f1"]),
            xytext=(8, 8),
            textcoords="offset points",
            color=config.colors.threshold,
            fontweight="bold",
        )
    _add_threshold_line(ax, threshold, config)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    if best_row is not None:
        ax.set_title(f"Threshold Sweep (best F1 = {best_row['f1']:.3f} at threshold = {best_row['threshold']:.3f})")
    else:
        ax.set_title("Threshold Sweep")
    ax.set_ylim(-0.03, 1.03)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_plots(
    df: pd.DataFrame,
    sweep: pd.DataFrame,
    output_dir: Path,
    config: PlotsConfig,
    threshold: float | str | None = None,
) -> dict[str, Path]:
    if not config.enabled:
        return {}
    paths: dict[str, Path] = {}
    if config.confusion_matrix.enabled:
        paths["confusion_matrix"] = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(df, paths["confusion_matrix"], config)
    if config.score_distribution.enabled:
        paths["score_distribution"] = output_dir / "score_distribution.png"
        plot_score_distribution(df, paths["score_distribution"], config, threshold)
    if config.per_image_score.enabled:
        paths["per_image_score"] = output_dir / "per_image_score.png"
        plot_per_image_score(df, paths["per_image_score"], config, threshold)
    if config.threshold_sweep.enabled:
        paths["threshold_sweep"] = output_dir / "threshold_sweep.png"
        plot_threshold_sweep(sweep, paths["threshold_sweep"], config, threshold)
    return paths
