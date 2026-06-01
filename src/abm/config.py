"""Configuration models and YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ALLOWED_CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth"}
REJECTED_CHECKPOINT_SUFFIXES = {".onnx", ".xml", ".bin", ".engine", ".trt", ".openvino"}


class ProjectConfig(BaseModel):
    name: str
    output_dir: Path


class EngineConfig(BaseModel):
    accelerator: str = "auto"
    devices: int | str | list[int] = 1
    logger: bool | Any = False
    enable_checkpointing: bool = True
    default_root_dir: Path | None = None
    init_args: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str | None = None
    class_path: str | None = None
    checkpoint: Path
    init_args: dict[str, Any] = Field(default_factory=dict)

    @field_validator("checkpoint")
    @classmethod
    def validate_checkpoint_suffix(cls, value: Path) -> Path:
        suffix = value.suffix.lower()
        if suffix in REJECTED_CHECKPOINT_SUFFIXES:
            raise ValueError(
                f"Unsupported checkpoint format '{suffix}'. v0.1 supports only "
                f"{sorted(ALLOWED_CHECKPOINT_SUFFIXES)} and does not support exported deployment formats."
            )
        if suffix not in ALLOWED_CHECKPOINT_SUFFIXES:
            raise ValueError(f"Unsupported checkpoint suffix '{suffix}'. Allowed: {sorted(ALLOWED_CHECKPOINT_SUFFIXES)}")
        return value

    @model_validator(mode="after")
    def require_name_or_path(self) -> "ModelConfig":
        if not self.name and not self.class_path:
            raise ValueError("model.name or model.class_path is required.")
        return self


class LabelFromPathConfig(BaseModel):
    normal_tokens: list[str] = Field(default_factory=lambda: ["good", "normal", "ok"])
    abnormal_tokens: list[str] = Field(default_factory=lambda: ["bad", "defect", "anomaly", "abnormal"])


class DatasetConfig(BaseModel):
    name: str | None = None
    class_path: str | None = None
    root: Path | None = None
    category: str | None = None
    normal_dir: str | None = None
    abnormal_dir: str | None = None
    normal_test_dir: str | None = None
    mask_dir: str | None = None
    eval_batch_size: int | None = 16
    num_workers: int | None = 0
    strict_args: bool = True
    label_from_path: LabelFromPathConfig = Field(default_factory=LabelFromPathConfig)
    init_args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_name_or_path(self) -> "DatasetConfig":
        if not self.name and not self.class_path:
            raise ValueError("dataset.name or dataset.class_path is required.")
        return self


class ThresholdConfig(BaseModel):
    method: Literal["best_f1", "manual", "use_model_threshold"] = "best_f1"
    manual_value: float | None = None
    num_thresholds: int = 1000

    @model_validator(mode="after")
    def validate_threshold(self) -> "ThresholdConfig":
        if self.method == "manual" and self.manual_value is None:
            raise ValueError("benchmark.threshold.manual_value is required when method is 'manual'.")
        if self.num_thresholds < 2:
            raise ValueError("benchmark.threshold.num_thresholds must be at least 2.")
        return self


class BenchmarkConfigSection(BaseModel):
    threshold: ThresholdConfig = Field(default_factory=ThresholdConfig)


class OutputsConfig(BaseModel):
    save_csv: bool = True
    save_plots: bool = True
    save_report_md: bool = True
    create_zip: bool = True


class PlotColorsConfig(BaseModel):
    good: str = "#ff7f0e"
    bad: str = "#1f77b4"
    threshold: str = "#d62728"
    correct: str = "#2ca02c"
    wrong: str = "#d62728"


class ScoreDistributionPlotConfig(BaseModel):
    enabled: bool = True
    kind: Literal["histogram"] = "histogram"
    bins: int = 40
    alpha: float = 0.55


class ConfidenceVsScorePlotConfig(BaseModel):
    enabled: bool = True
    kind: Literal["scatter"] = "scatter"
    alpha: float = 0.65
    marker_size: int = 35


class ConfusionMatrixPlotConfig(BaseModel):
    enabled: bool = True
    show_values: bool = True
    colorbar: bool = True


class ThresholdSweepPlotConfig(BaseModel):
    enabled: bool = True
    kind: Literal["line"] = "line"


class PlotsConfig(BaseModel):
    enabled: bool = True
    dpi: int = 200
    figure_size: tuple[float, float] = (9, 6)
    theme: Literal["default"] = "default"
    colors: PlotColorsConfig = Field(default_factory=PlotColorsConfig)
    score_distribution: ScoreDistributionPlotConfig = Field(default_factory=ScoreDistributionPlotConfig)
    confidence_vs_score: ConfidenceVsScorePlotConfig = Field(default_factory=ConfidenceVsScorePlotConfig)
    confusion_matrix: ConfusionMatrixPlotConfig = Field(default_factory=ConfusionMatrixPlotConfig)
    threshold_sweep: ThresholdSweepPlotConfig = Field(default_factory=ThresholdSweepPlotConfig)

    @field_validator("figure_size", mode="before")
    @classmethod
    def validate_figure_size(cls, value: Any) -> tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise ValueError("plots.figure_size must be a two-item list, for example [9, 6].")


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    project: ProjectConfig
    engine: EngineConfig = Field(default_factory=EngineConfig)
    model: ModelConfig
    dataset: DatasetConfig
    benchmark: BenchmarkConfigSection = Field(default_factory=BenchmarkConfigSection)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    plots: PlotsConfig = Field(default_factory=PlotsConfig)


class CliOverrides(BaseModel):
    model: str | None = None
    dataset: str | None = None
    category: str | None = None
    root: Path | None = None
    ckpt: Path | None = None
    output: Path | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must contain a YAML mapping.")
    return data


def merge_cli_overrides(data: dict[str, Any], overrides: CliOverrides | None) -> dict[str, Any]:
    if overrides is None:
        return data
    merged = dict(data)
    if overrides.output is not None:
        merged.setdefault("project", {})["output_dir"] = overrides.output
    if overrides.model is not None:
        merged.setdefault("model", {})["name"] = overrides.model
        merged["model"]["class_path"] = None
    if overrides.ckpt is not None:
        merged.setdefault("model", {})["checkpoint"] = overrides.ckpt
    if overrides.dataset is not None:
        merged.setdefault("dataset", {})["name"] = overrides.dataset
        merged["dataset"]["class_path"] = None
    if overrides.category is not None:
        merged.setdefault("dataset", {})["category"] = overrides.category
    if overrides.root is not None:
        merged.setdefault("dataset", {})["root"] = overrides.root
    return merged


def load_config(path: Path, overrides: CliOverrides | None = None) -> BenchmarkConfig:
    data = merge_cli_overrides(load_yaml(path), overrides)
    return BenchmarkConfig.model_validate(data)
