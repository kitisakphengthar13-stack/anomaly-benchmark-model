"""Shared utility helpers."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


class BenchmarkError(RuntimeError):
    """Base error for actionable benchmark failures."""


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_object(class_path: str) -> Any:
    """Import an object from a dotted class path."""
    if "." not in class_path:
        raise BenchmarkError(f"Invalid class path '{class_path}'. Expected 'module.ClassName'.")
    module_name, attr_name = class_path.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except Exception as exc:  # pragma: no cover - exact import errors depend on environment
        raise BenchmarkError(f"Could not import module '{module_name}' from '{class_path}': {exc}") from exc
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise BenchmarkError(f"Module '{module_name}' does not define '{attr_name}'.") from exc
