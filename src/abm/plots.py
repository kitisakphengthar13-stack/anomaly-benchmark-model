"""Matplotlib plot generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import PlotsConfig
from .metrics import confusion_matrix_table


def _figure(config: PlotsConfig):
    return plt.subplots(figsize=config.figure_size, dpi=config.dpi)


def plot_confusion_matrix(df: pd.DataFrame, output_path: Path, config: PlotsConfig) -> None:
    table = confusion_matrix_table(df).iloc[0]
    matrix = [[table["tn"], table["fp"]], [table["fn"], table["tp"]]]
    fig, ax = _figure(config)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred good", "Pred bad"])
    ax.set_yticks([0, 1], labels=["True good", "True bad"])
    ax.set_title("Confusion Matrix")
    if config.confusion_matrix.show_values:
        for y in range(2):
            for x in range(2):
                ax.text(x, y, str(matrix[y][x]), ha="center", va="center", color="black")
    if config.confusion_matrix.colorbar:
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_score_distribution(df: pd.DataFrame, output_path: Path, config: PlotsConfig) -> None:
    fig, ax = _figure(config)
    good = df[df["y_true"] == 0]["score"].dropna()
    bad = df[df["y_true"] == 1]["score"].dropna()
    ax.hist(good, bins=config.score_distribution.bins, alpha=config.score_distribution.alpha, label="good", color=config.colors.good)
    ax.hist(bad, bins=config.score_distribution.bins, alpha=config.score_distribution.alpha, label="bad", color=config.colors.bad)
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Image count")
    ax.set_title("Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confidence_vs_score(df: pd.DataFrame, output_path: Path, config: PlotsConfig) -> None:
    plot_df = df.dropna(subset=["score", "y_true"]).copy()
    fig, ax = _figure(config)
    colors = plot_df["y_true"].map({0: config.colors.good, 1: config.colors.bad}).fillna("#777777")
    x_values = range(len(plot_df))
    ax.scatter(x_values, plot_df["score"], c=colors, alpha=config.confidence_vs_score.alpha, s=config.confidence_vs_score.marker_size)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Anomaly score")
    ax.set_title("Confidence vs Score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_threshold_sweep(sweep: pd.DataFrame, output_path: Path, config: PlotsConfig) -> None:
    fig, ax = _figure(config)
    if not sweep.empty:
        ax.plot(sweep["threshold"], sweep["precision"], label="precision")
        ax.plot(sweep["threshold"], sweep["recall"], label="recall")
        ax.plot(sweep["threshold"], sweep["f1"], label="f1", color=config.colors.threshold)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold Sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_plots(df: pd.DataFrame, sweep: pd.DataFrame, output_dir: Path, config: PlotsConfig) -> dict[str, Path]:
    if not config.enabled:
        return {}
    paths: dict[str, Path] = {}
    if config.confusion_matrix.enabled:
        paths["confusion_matrix"] = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(df, paths["confusion_matrix"], config)
    if config.score_distribution.enabled:
        paths["score_distribution"] = output_dir / "score_distribution.png"
        plot_score_distribution(df, paths["score_distribution"], config)
    if config.confidence_vs_score.enabled:
        paths["confidence_vs_score"] = output_dir / "confidence_vs_score.png"
        plot_confidence_vs_score(df, paths["confidence_vs_score"], config)
    if config.threshold_sweep.enabled:
        paths["threshold_sweep"] = output_dir / "threshold_sweep.png"
        plot_threshold_sweep(sweep, paths["threshold_sweep"], config)
    return paths
