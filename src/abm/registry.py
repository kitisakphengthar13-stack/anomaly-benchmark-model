"""Lazy registries for Anomalib models and datamodules."""

from __future__ import annotations

from typing import Mapping

from .utils import BenchmarkError, import_object


MODEL_REGISTRY: dict[str, str] = {
    "AiVad": "anomalib.models.AiVad",
    "Cfa": "anomalib.models.Cfa",
    "Cflow": "anomalib.models.Cflow",
    "Csflow": "anomalib.models.Csflow",
    "Dfkde": "anomalib.models.Dfkde",
    "Dfm": "anomalib.models.Dfm",
    "Draem": "anomalib.models.Draem",
    "Dsr": "anomalib.models.Dsr",
    "EfficientAd": "anomalib.models.EfficientAd",
    "Fastflow": "anomalib.models.Fastflow",
    "Fre": "anomalib.models.Fre",
    "Ganomaly": "anomalib.models.Ganomaly",
    "Padim": "anomalib.models.Padim",
    "Patchcore": "anomalib.models.Patchcore",
    "ReverseDistillation": "anomalib.models.ReverseDistillation",
    "Stfpm": "anomalib.models.Stfpm",
    "Uflow": "anomalib.models.Uflow",
    "WinClip": "anomalib.models.WinClip",
    "Dinomaly": "anomalib.models.Dinomaly",
    "Dinomaly2": "anomalib.models.Dinomaly2",
    "PatchFlow": "anomalib.models.PatchFlow",
    "GeneralAd": "anomalib.models.GeneralAd",
    "L2Bt": "anomalib.models.L2Bt",
}

MODEL_ALIASES: dict[str, str] = {
    "aivad": "AiVad",
    "cfa": "Cfa",
    "cflow": "Cflow",
    "csflow": "Csflow",
    "dfkde": "Dfkde",
    "dfm": "Dfm",
    "draem": "Draem",
    "dsr": "Dsr",
    "efficientad": "EfficientAd",
    "efficient_ad": "EfficientAd",
    "fastflow": "Fastflow",
    "fre": "Fre",
    "ganomaly": "Ganomaly",
    "padim": "Padim",
    "patchcore": "Patchcore",
    "patch_core": "Patchcore",
    "rd": "ReverseDistillation",
    "reverse_distillation": "ReverseDistillation",
    "reversedistillation": "ReverseDistillation",
    "stfpm": "Stfpm",
    "uflow": "Uflow",
    "winclip": "WinClip",
    "win_clip": "WinClip",
    "dinomaly": "Dinomaly",
    "dinomaly2": "Dinomaly2",
    "patchflow": "PatchFlow",
    "patch_flow": "PatchFlow",
    "generalad": "GeneralAd",
    "general_ad": "GeneralAd",
    "l2bt": "L2Bt",
}

DATASET_REGISTRY: dict[str, str] = {
    "Folder": "anomalib.data.Folder",
    "Visa": "anomalib.data.Visa",
    "MVTecAD": "anomalib.data.MVTecAD",
    "MVTecAD2": "anomalib.data.MVTecAD2",
    "MVTecLOCO": "anomalib.data.MVTecLOCO",
    "BTech": "anomalib.data.BTech",
    "Kolektor": "anomalib.data.Kolektor",
    "DAGM": "anomalib.data.DAGM",
    "RealIAD": "anomalib.data.RealIAD",
    "MPDD": "anomalib.data.MPDD",
    "BTAD": "anomalib.data.BTAD",
    "BeanTech": "anomalib.data.BeanTech",
    "Datumaro": "anomalib.data.Datumaro",
    "VAD": "anomalib.data.VAD",
    "Kaputt": "anomalib.data.Kaputt",
}

DATASET_ALIASES: dict[str, str] = {
    "folder": "Folder",
    "custom": "Folder",
    "visa": "Visa",
    "mvtecad": "MVTecAD",
    "mvtec": "MVTecAD",
    "mvtecad2": "MVTecAD2",
    "mvtecloco": "MVTecLOCO",
    "mvtec_loco": "MVTecLOCO",
    "btech": "BTech",
    "kolektor": "Kolektor",
    "dagm": "DAGM",
    "realiad": "RealIAD",
    "real_iad": "RealIAD",
    "mpdd": "MPDD",
    "btad": "BTAD",
    "beantech": "BeanTech",
    "datumaro": "Datumaro",
    "vad": "VAD",
    "kaputt": "Kaputt",
}


def import_class(class_path: str):
    """Import a class by dotted path."""
    return import_object(class_path)


def canonical_name(name: str, registry: Mapping[str, str], aliases: Mapping[str, str]) -> str:
    if name in registry:
        return name
    alias_key = name.strip().lower().replace("-", "_")
    if alias_key in aliases:
        return aliases[alias_key]
    raise BenchmarkError(
        f"Unknown registry name '{name}'. Registered names: {', '.join(sorted(registry))}. "
        "Use class_path for custom or newly added Anomalib classes."
    )


def resolve_class(
    *,
    name: str | None,
    class_path: str | None,
    registry: Mapping[str, str],
    aliases: Mapping[str, str],
    kind: str,
):
    """Resolve a class using class_path first, then registry name/alias."""
    if class_path:
        return import_class(class_path)
    if not name:
        raise BenchmarkError(f"{kind}.name or {kind}.class_path is required.")
    canonical = canonical_name(name, registry, aliases)
    try:
        return import_class(registry[canonical])
    except BenchmarkError as exc:
        raise BenchmarkError(
            f"Could not import {kind} '{canonical}' from '{registry[canonical]}'. "
            f"It may not exist in the installed Anomalib version. Use {kind}.class_path if needed. "
            f"Original error: {exc}"
        ) from exc


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def list_datasets() -> list[str]:
    return sorted(DATASET_REGISTRY)
