"""Anomalib datamodule factory."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

from .config import DatasetConfig
from .registry import DATASET_ALIASES, DATASET_REGISTRY, resolve_class
from .utils import BenchmarkError


def _candidate_dataset_kwargs(config: DatasetConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in [
        "root",
        "category",
        "normal_dir",
        "abnormal_dir",
        "normal_test_dir",
        "mask_dir",
        "eval_batch_size",
        "num_workers",
    ]:
        value = getattr(config, key)
        if value is not None:
            kwargs[key] = value
    kwargs.update(config.init_args)
    return kwargs


def filter_kwargs_for_signature(cls: type, kwargs: dict[str, Any], *, strict: bool) -> dict[str, Any]:
    """Filter keyword arguments according to a constructor signature."""
    signature = inspect.signature(cls.__init__)
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs
    allowed = {
        name
        for name, param in params.items()
        if name != "self" and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    unsupported = sorted(set(kwargs) - allowed)
    if unsupported and strict:
        raise BenchmarkError(
            f"Unsupported dataset args for {cls.__module__}.{cls.__name__}: {unsupported}. "
            "Move unsupported values out of dataset.init_args or set dataset.strict_args=false to skip them."
        )
    if unsupported:
        warnings.warn(
            f"Skipping unsupported dataset args for {cls.__module__}.{cls.__name__}: {unsupported}",
            stacklevel=2,
        )
    return {key: value for key, value in kwargs.items() if key in allowed}


def create_datamodule(config: DatasetConfig):
    """Instantiate an Anomalib datamodule from config."""
    dataset_cls = resolve_class(
        name=config.name,
        class_path=config.class_path,
        registry=DATASET_REGISTRY,
        aliases=DATASET_ALIASES,
        kind="dataset",
    )
    kwargs = filter_kwargs_for_signature(dataset_cls, _candidate_dataset_kwargs(config), strict=config.strict_args)
    return dataset_cls(**kwargs)
