"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import CliOverrides, load_config
from .registry import list_datasets, list_models
from .runner import run_benchmark
from .utils import BenchmarkError


app = typer.Typer(help="CLI benchmark/reporting tool for Anomalib models.")
console = Console()


@app.command("run")
def run_command(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config."),
    model: str | None = typer.Option(None, "--model", help="Override model.name."),
    dataset: str | None = typer.Option(None, "--dataset", help="Override dataset.name."),
    category: str | None = typer.Option(None, "--category", help="Override dataset.category."),
    root: Path | None = typer.Option(None, "--root", help="Override dataset.root."),
    ckpt: Path | None = typer.Option(None, "--ckpt", help="Override model.checkpoint."),
    output: Path | None = typer.Option(None, "--output", help="Override project.output_dir."),
) -> None:
    """Run one benchmark."""
    try:
        cfg = load_config(config, CliOverrides(model=model, dataset=dataset, category=category, root=root, ckpt=ckpt, output=output))
        result = run_benchmark(cfg)
    except Exception as exc:
        message = str(exc)
        if not isinstance(exc, BenchmarkError):
            message = f"{type(exc).__name__}: {message}"
        console.print(f"[red]Error:[/red] {message}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Benchmark complete.[/green] Output: {result.output_dir}")


@app.command("validate-config")
def validate_config(config: Path = typer.Option(..., "--config", "-c", help="Path to YAML config.")) -> None:
    """Validate a YAML config without running Anomalib."""
    try:
        cfg = load_config(config)
    except Exception as exc:
        console.print(f"[red]Invalid config:[/red] {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    console.print(f"[green]Valid config:[/green] {cfg.project.name}")


@app.command("list-models")
def list_models_command() -> None:
    """List registered model names."""
    table = Table(title="Registered Models")
    table.add_column("Name")
    for name in list_models():
        table.add_row(name)
    console.print(table)


@app.command("list-datasets")
def list_datasets_command() -> None:
    """List registered dataset names."""
    table = Table(title="Registered Datasets")
    table.add_column("Name")
    for name in list_datasets():
        table.add_row(name)
    console.print(table)


if __name__ == "__main__":
    app()
