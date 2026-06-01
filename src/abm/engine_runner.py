"""Thin Anomalib Engine wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from .config import BenchmarkConfig
from .utils import BenchmarkError


@dataclass
class EngineRunResult:
    aggregate_metrics: list[dict]
    raw_predictions: object


def create_engine(config: BenchmarkConfig):
    try:
        from anomalib.engine import Engine
    except Exception as exc:  # pragma: no cover - depends on environment
        raise BenchmarkError(f"Could not import anomalib.engine.Engine: {exc}") from exc

    kwargs = {
        "accelerator": config.engine.accelerator,
        "devices": config.engine.devices,
        "logger": config.engine.logger,
        "enable_checkpointing": config.engine.enable_checkpointing,
    }
    if config.engine.default_root_dir is not None:
        kwargs["default_root_dir"] = config.engine.default_root_dir
    kwargs.update(config.engine.init_args)
    return Engine(**kwargs)


def run_engine(config: BenchmarkConfig, model, datamodule) -> EngineRunResult:
    """Run Anomalib test and predict without custom inference logic."""
    engine = create_engine(config)
    ckpt_path = str(config.model.checkpoint)
    try:
        aggregate_metrics = engine.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, verbose=True)
        raw_predictions = engine.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            return_predictions=True,
        )
    except Exception as exc:  # pragma: no cover - requires Anomalib runtime
        raise BenchmarkError(
            "Anomalib Engine failed while using the configured checkpoint. "
            "Use a Lightning-compatible .ckpt when possible. For .pt/.pth files, support depends on "
            f"the installed Anomalib/Lightning model loader. Original error: {exc}"
        ) from exc
    return EngineRunResult(aggregate_metrics=aggregate_metrics or [], raw_predictions=raw_predictions)
