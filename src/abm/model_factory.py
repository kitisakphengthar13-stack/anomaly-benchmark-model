"""Anomalib model factory."""

from __future__ import annotations

from .config import ModelConfig
from .registry import MODEL_ALIASES, MODEL_REGISTRY, resolve_class


def create_model(config: ModelConfig):
    """Instantiate an Anomalib model from config."""
    model_cls = resolve_class(
        name=config.name,
        class_path=config.class_path,
        registry=MODEL_REGISTRY,
        aliases=MODEL_ALIASES,
        kind="model",
    )
    return model_cls(**config.init_args)
