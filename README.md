# anomaly-benchmark-model

`anomaly-benchmark-model` is a CLI-first, YAML-config-driven benchmark and reporting layer for Anomalib anomaly detection models.

It does not replace Anomalib. Anomalib remains responsible for model construction, datamodules, testing, prediction, post-processing, and checkpoint loading. This project turns Anomalib test metrics and prediction outputs into reproducible CSV files, diagnostic plots, and a human-readable Markdown report.

## What it does

- Runs one model checkpoint against one Anomalib datamodule.
- Uses `Engine.test()` for aggregate metrics.
- Uses `Engine.predict(..., return_predictions=True)` for prediction objects.
- Exports per-image scores, threshold analysis, confusion matrix, FP/FN lists, defect breakdowns, plots, and `report.md`.
- Supports config-first workflows for reproducible benchmarks.

## What it does not do

- No GUI.
- No training workflow.
- No custom model inference implementation.
- No ONNX, OpenVINO, TensorRT, IR, or remote serving support in v0.1.
- No batch model comparison in v0.1.

## Recommended workflow

From the project root, the folder containing `pyproject.toml`, install the package in editable mode:

```powershell
python -m pip install -e .
```

Then run the CLI:

```powershell
abm --help
abm validate-config --config configs/rd_visa_capsules.yaml
abm list-models
abm list-datasets
abm run --config configs/rd_visa_capsules.yaml
```

The full CLI name is also available:

```powershell
anomaly-benchmark-model --help
anomaly-benchmark-model run --config configs/rd_visa_capsules.yaml
```

`abm` and `anomaly-benchmark-model` are console scripts registered by `pyproject.toml`. They become available after `python -m pip install -e .`.

Editable install means changes under `src/abm/` are picked up without reinstalling the package. This is the recommended workflow for development and normal use.

Anomalib 2.5.0 and its backend dependencies must be available in the active Python environment.

## CLI

The main command is:

```powershell
abm run --config configs/rd_visa_capsules.yaml
```

Limited overrides are supported:

```powershell
abm run `
  --config configs/rd_visa_capsules.yaml `
  --model ReverseDistillation `
  --dataset Visa `
  --category capsules `
  --root D:/datasets/Visa `
  --ckpt D:/models/model.ckpt `
  --output outputs/run1
```

Other commands:

```powershell
abm validate-config --config configs/rd_visa_capsules.yaml
abm list-models
abm list-datasets
```

## Running without installation

This is optional and mainly useful for development or debugging. The intended public package name is `abm`.

PowerShell:

```powershell
$env:PYTHONPATH="src"
python -m abm.cli --help
python -m abm.cli validate-config --config configs/rd_visa_capsules.yaml
python -m abm.cli run --config configs/rd_visa_capsules.yaml
```

CMD:

```cmd
set PYTHONPATH=src
python -m abm.cli --help
python -m abm.cli run --config configs/rd_visa_capsules.yaml
```

Bash/Linux/macOS:

```bash
export PYTHONPATH=src
python -m abm.cli --help
python -m abm.cli run --config configs/rd_visa_capsules.yaml
```

Do not use `python -m src.abm.cli` as the public usage pattern. It may work in some local source-tree situations, but the intended public package name is `abm`. Prefer either editable install plus `abm`, or `PYTHONPATH=src python -m abm.cli`.

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
- `confidence_vs_score.png`
- `threshold_sweep.png`
- `report.md`
- optional zip archive

## Checkpoints

v0.1 accepts PyTorch/Lightning-family checkpoint suffixes:

- `.ckpt`
- `.pt`
- `.pth`

The recommended format is Lightning `.ckpt`. `.pt` and `.pth` paths are passed as `ckpt_path` to Anomalib/Lightning. If the backend does not support the file for a specific model, the run fails with an actionable error. This project does not implement custom `state_dict` loading in v0.1.

## Custom datasets

Use Anomalib `Folder` for simple custom datasets:

```yaml
dataset:
  name: Folder
  root: D:/datasets/custom_product
  category: null
  normal_dir: good
  abnormal_dir: bad
  normal_test_dir: null
  mask_dir: masks
  eval_batch_size: 16
  num_workers: 0
  strict_args: true
  init_args: {}
```

Custom datamodules are supported by class path:

```yaml
dataset:
  class_path: my_package.my_dataset.MyAnomalibDataModule
  root: D:/datasets/custom
  init_args:
    some_arg: value
```

## Dynamic model class path

```yaml
model:
  name: null
  class_path: anomalib.models.Padim
  checkpoint: D:/models/padim/model.ckpt
  init_args: {}
```

## Compatibility notes

Prediction object shape can differ across Anomalib models, tasks, and datamodules. The parser is intentionally defensive and supports object predictions, dictionaries, batched values, tensors, NumPy arrays, scalars, paths, bytes, and nested one-item lists. Full `anomaly_map` and `pred_mask` arrays are not written to CSV in v0.1.

## Troubleshooting

### `abm` command not found

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

### Pytest temp directory issue on Windows

If pytest cannot access the default temp directory, run:

```powershell
python -m pytest --basetemp .pytest_tmp
```
