"""File exporters."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def save_csv_outputs(
    output_dir: Path,
    *,
    benchmark_metrics: pd.DataFrame,
    per_image: pd.DataFrame,
    per_image_with_threshold: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
    confusion_matrix: pd.DataFrame,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
    defect_breakdown: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "benchmark_metrics": (benchmark_metrics, output_dir / "benchmark_metrics.csv"),
        "per_image_scores": (per_image, output_dir / "per_image_scores.csv"),
        "per_image_scores_with_threshold": (per_image_with_threshold, output_dir / "per_image_scores_with_threshold.csv"),
        "threshold_summary": (threshold_summary, output_dir / "threshold_summary.csv"),
        "threshold_sweep": (threshold_sweep, output_dir / "threshold_sweep.csv"),
        "confusion_matrix": (confusion_matrix, output_dir / "confusion_matrix.csv"),
        "false_positive_list": (false_positives, output_dir / "false_positive_list.csv"),
        "false_negative_list": (false_negatives, output_dir / "false_negative_list.csv"),
        "defect_type_breakdown": (defect_breakdown, output_dir / "defect_type_breakdown.csv"),
    }
    paths = {}
    for name, (frame, path) in outputs.items():
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def save_report(output_dir: Path, markdown: str) -> Path:
    path = output_dir / "report.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def create_zip_archive(output_dir: Path, project_name: str) -> Path:
    zip_path = output_dir / f"{project_name}.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in output_dir.iterdir():
            if path == zip_path or not path.is_file():
                continue
            archive.write(path, arcname=path.name)
    return zip_path
