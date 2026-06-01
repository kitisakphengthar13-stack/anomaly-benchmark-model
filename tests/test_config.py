from pathlib import Path

import pytest

from abm.config import BenchmarkConfig, CliOverrides, load_config, merge_cli_overrides


def minimal_config(checkpoint: str = "D:/models/model.ckpt") -> dict:
    return {
        "project": {"name": "x", "output_dir": "outputs/x"},
        "model": {"name": "Patchcore", "checkpoint": checkpoint, "init_args": {}},
        "dataset": {"name": "Folder", "root": "D:/data", "init_args": {}},
    }


def test_config_accepts_windows_checkpoint_path():
    cfg = BenchmarkConfig.model_validate(minimal_config("D:/models/model.ckpt"))
    assert cfg.model.checkpoint == Path("D:/models/model.ckpt")


def test_config_rejects_exported_formats():
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        BenchmarkConfig.model_validate(minimal_config("D:/models/model.onnx"))


def test_config_rejects_unknown_suffix():
    with pytest.raises(ValueError, match="Unsupported checkpoint suffix"):
        BenchmarkConfig.model_validate(minimal_config("D:/models/model.txt"))


def test_cli_overrides_merge():
    data = minimal_config()
    merged = merge_cli_overrides(
        data,
        CliOverrides(
            model="ReverseDistillation",
            dataset="Visa",
            category="capsules",
            root=Path("D:/datasets/Visa"),
            ckpt=Path("D:/models/rd.ckpt"),
            output=Path("outputs/run1"),
        ),
    )
    assert merged["model"]["name"] == "ReverseDistillation"
    assert merged["dataset"]["name"] == "Visa"
    assert merged["dataset"]["category"] == "capsules"
    assert merged["project"]["output_dir"] == Path("outputs/run1")


def test_load_example_config():
    cfg = load_config(Path("configs/rd_visa_capsules.yaml"))
    assert cfg.project.name == "rd_visa_capsules"
    assert cfg.model.name == "ReverseDistillation"
