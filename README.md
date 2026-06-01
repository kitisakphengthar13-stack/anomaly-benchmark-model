# anomaly-benchmark-model

`anomaly-benchmark-model` is a CLI-first, YAML-config-driven benchmark and reporting layer for Anomalib anomaly detection models.

It does not replace Anomalib. Anomalib remains responsible for model construction, datamodules, testing, prediction, post-processing, and checkpoint loading. This project turns Anomalib test metrics and prediction outputs into reproducible CSV files, diagnostic plots, and a human-readable Markdown report.

## What It Does

- Runs one model checkpoint against one Anomalib datamodule.
- Uses `Engine.test()` for aggregate metrics.
- Uses `Engine.predict(..., return_predictions=True)` for prediction objects.
- Exports per-image scores, threshold analysis, confusion matrix, FP/FN lists, defect breakdowns, plots, and `report.md`.
- Supports config-first workflows for reproducible benchmarks.

## What It Does Not Do

- No GUI.
- No training workflow.
- No custom model inference implementation.
- No ONNX, OpenVINO, TensorRT, IR, or remote serving support in v0.1.
- No batch model comparison in v0.1.

## Recommended Workflow

From the project root, the folder containing `pyproject.toml`, install the package in editable mode:

```powershell
python -m pip install -e .
```

Then run the CLI:

```powershell
abm --help
abm validate-config --config configs/anomalib_dataset_example.yaml
abm list-models
abm list-datasets
```

`abm` and `anomaly-benchmark-model` are console scripts registered by `pyproject.toml`. They become available after `python -m pip install -e .`.

Editable install means changes under `src/abm/` are picked up without reinstalling the package. This is the recommended workflow for development and normal use.

The full CLI name is also available:

```powershell
anomaly-benchmark-model --help
```

Anomalib 2.5.0 and its backend dependencies must be available in the active Python environment.

## Config Policy

There are exactly two official tracked example YAML files:

- `configs/anomalib_dataset_example.yaml`
- `configs/custom_folder_dataset_example.yaml`

Use `configs/anomalib_dataset_example.yaml` for Anomalib built-in datamodules such as `MVTecAD`, `Visa`, `BTech`, `Kolektor`, `MPDD`, `RealIAD`, `MVTecLOCO`, `MVTecAD2`, `Datumaro`, or `VAD`.

Use `configs/custom_folder_dataset_example.yaml` for user-owned folder datasets through Anomalib `Folder`.

Real local configs are ignored by Git because they usually contain local dataset paths, checkpoint paths, and output paths. Local config names can include:

- `configs/cable_patchcore.yaml`
- `configs/transistor_patchcore.yaml`
- `configs/local/my_private_config.yaml`
- `configs/my_experiment.local.yaml`

These local configs will not be committed.

## Quick Start

For a built-in Anomalib dataset:

```powershell
copy configs/anomalib_dataset_example.yaml configs/cable_patchcore.yaml
abm validate-config --config configs/cable_patchcore.yaml
abm run --config configs/cable_patchcore.yaml
```

For a custom folder dataset:

```powershell
copy configs/custom_folder_dataset_example.yaml configs/my_product_patchcore.yaml
abm validate-config --config configs/my_product_patchcore.yaml
abm run --config configs/my_product_patchcore.yaml
```

After copying an example, edit:

- `project.name`
- `project.output_dir`
- `model.name` or `model.class_path`
- `model.checkpoint`
- `dataset.name` or `dataset.class_path`
- `dataset.root`
- `dataset.category` for built-in datasets
- `dataset.normal_dir`, `dataset.abnormal_dir`, and `dataset.mask_dir` for `Folder`

Validate official examples with placeholder paths:

```powershell
abm validate-config --config configs/anomalib_dataset_example.yaml
abm validate-config --config configs/custom_folder_dataset_example.yaml
```

## CLI

Limited overrides are supported:

```powershell
abm run `
  --config configs/cable_patchcore.yaml `
  --model Patchcore `
  --dataset MVTecAD `
  --category cable `
  --root D:/path/to/MVTecAD `
  --ckpt D:/path/to/model.ckpt `
  --output outputs/cable_patchcore
```

Other commands:

```powershell
abm validate-config --config configs/anomalib_dataset_example.yaml
abm list-models
abm list-datasets
```

## Running Without Installation

This is optional and mainly useful for development or debugging. The intended public package name is `abm`.

PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m abm.cli --help
python -m abm.cli validate-config --config configs/anomalib_dataset_example.yaml
python -m abm.cli run --config configs/cable_patchcore.yaml
```

CMD:

```cmd
set PYTHONPATH=src
python -m abm.cli --help
python -m abm.cli run --config configs/cable_patchcore.yaml
```

Bash/Linux/macOS:

```bash
export PYTHONPATH=src
python -m abm.cli --help
python -m abm.cli run --config configs/cable_patchcore.yaml
```

Do not use `python -m src.abm.cli` as the public usage pattern. It may work in some local source-tree situations, but the intended public package name is `abm`. Prefer either editable install plus `abm`, or `PYTHONPATH=src python -m abm.cli`.

## Anomalib Support

This tool supports Anomalib models through registry names and `model.class_path`. It supports Anomalib datamodules through registry names and `dataset.class_path`. For custom image datasets, use Anomalib `Folder`.

The exact available classes depend on the installed Anomalib version. If a model or dataset name is not available locally, use `class_path` or upgrade Anomalib.

Current Anomalib documentation lists image models such as AnomalyVFM, AnomalyDINO, CFA, CFM, C-Flow, CS-Flow, DFKDE, DFM, Dinomaly, DRAEM, DSR, EfficientAD, INP-Former, FastFlow, FRE, GANomaly, GLASS, GeneralAD, L2BT, PaDiM, Patchcore, PatchFlow, Reverse Distillation, STFPM, SuperSimpleNet, U-Flow, UniNet, VLM-AD, and WinCLIP.

Current image datamodule documentation lists BMAD, BTech, Datumaro, Folder, Kolektor, MPDD, MVTecAD, MVTecAD2, MVTecLOCO, RealIAD, Tabular, VAD, and Visa.

This project resolves and runs Anomalib classes, but it does not guarantee every class/checkpoint/config combination is valid. `model.init_args` must match how the checkpoint was trained, and dataset config fields must match the selected datamodule constructor.

## Outputs

The default run creates:

- `benchmark_metrics.csv`
- `per_image_scores.csv`
- `per_image_scores_with_threshold.csv`
- `threshold_summary.csv`
- `threshold_sweep.csv`
- `confusion_matrix.csv`
- `false_positive_list.csv`
- `false_negative_list.csv`
- `defect_type_breakdown.csv`
- `confusion_matrix.png`
- `score_distribution.png`
- `per_image_score.png`
- `threshold_sweep.png`
- `report.md`
- optional zip archive

## Checkpoints

v0.1 accepts PyTorch/Lightning-family checkpoint suffixes:

- `.ckpt`
- `.pt`
- `.pth`

The recommended format is Lightning `.ckpt`. `.pt` and `.pth` paths are passed as `ckpt_path` to Anomalib/Lightning. If the backend does not support the file for a specific model, the run fails with an actionable error. This project does not implement custom `state_dict` loading in v0.1.

## Compatibility Notes

Prediction object shape can differ across Anomalib models, tasks, and datamodules. The parser is intentionally defensive and supports object predictions, dictionaries, batched values, tensors, NumPy arrays, scalars, paths, bytes, and nested one-item lists. Full `anomaly_map` and `pred_mask` arrays are not written to CSV in v0.1.

## Troubleshooting

### `abm` Command Not Found

Install the package from the project root:

```powershell
python -m pip install -e .
```

Then verify which executables are active:

```powershell
where python
where pip
where abm
```

Command-not-found usually means the package is not installed in the active Python environment, or the wrong Python/virtual environment is active.

### Pytest Temp Directory Issue On Windows

If pytest cannot access the default temp directory, run:

```powershell
python -m pytest --basetemp .pytest_tmp
```
