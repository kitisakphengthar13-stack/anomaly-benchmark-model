"""Benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import BenchmarkConfig
from .data_factory import create_datamodule
from .engine_runner import run_engine
from .exporters import create_zip_archive, save_csv_outputs, save_report
from .metrics import confusion_matrix_table, defect_type_breakdown, false_negative_table, false_positive_table, scalar_metrics
from .model_factory import create_model
from .plots import save_plots
from .prediction_parser import parse_predictions
from .reports import flatten_aggregate_metrics, generate_markdown_report
from .thresholding import apply_threshold
from .utils import ensure_dir


@dataclass
class BenchmarkResult:
    output_dir: Path
    csv_paths: dict[str, Path] = field(default_factory=dict)
    plot_paths: dict[str, Path] = field(default_factory=dict)
    report_path: Path | None = None
    zip_path: Path | None = None


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    output_dir = ensure_dir(config.project.output_dir)
    model = create_model(config.model)
    datamodule = create_datamodule(config.dataset)
    engine_result = run_engine(config, model, datamodule)

    model_name = config.model.name or config.model.class_path or type(model).__name__
    dataset_name = config.dataset.name or config.dataset.class_path or type(datamodule).__name__
    per_image = parse_predictions(
        engine_result.raw_predictions,
        model_name=model_name,
        dataset_name=dataset_name,
        category=config.dataset.category,
        label_from_path=config.dataset.label_from_path,
    )
    threshold_result = apply_threshold(per_image, config.benchmark.threshold)
    per_image_with_threshold = threshold_result.per_image
    benchmark_metrics = flatten_aggregate_metrics(engine_result.aggregate_metrics)
    cm = confusion_matrix_table(per_image_with_threshold)
    fp = false_positive_table(per_image_with_threshold)
    fn = false_negative_table(per_image_with_threshold)
    breakdown = defect_type_breakdown(per_image_with_threshold)
    scalars = scalar_metrics(per_image_with_threshold)

    result = BenchmarkResult(output_dir=output_dir)
    if config.outputs.save_csv:
        result.csv_paths = save_csv_outputs(
            output_dir,
            benchmark_metrics=benchmark_metrics,
            per_image=per_image,
            per_image_with_threshold=per_image_with_threshold,
            threshold_summary=threshold_result.summary,
            threshold_sweep=threshold_result.sweep,
            confusion_matrix=cm,
            false_positives=fp,
            false_negatives=fn,
            defect_breakdown=breakdown,
        )
    if config.outputs.save_plots:
        result.plot_paths = save_plots(
            per_image_with_threshold,
            threshold_result.sweep,
            output_dir,
            config.plots,
            threshold_result.threshold,
        )
    if config.outputs.save_report_md:
        markdown = generate_markdown_report(
            project_name=config.project.name,
            model_name=model_name,
            dataset_name=dataset_name,
            category=config.dataset.category,
            checkpoint=str(config.model.checkpoint),
            aggregate_metrics=benchmark_metrics,
            threshold_summary=threshold_result.summary,
            confusion_matrix=cm,
            scalar_metrics=scalars,
            false_positives=fp,
            false_negatives=fn,
            defect_breakdown=breakdown,
        )
        result.report_path = save_report(output_dir, markdown)
    if config.outputs.create_zip:
        result.zip_path = create_zip_archive(output_dir, config.project.name)
    return result
