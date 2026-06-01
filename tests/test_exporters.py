import pandas as pd

from abm.exporters import create_zip_archive, save_csv_outputs, save_report


def test_exporters_create_files(tmp_path):
    frame = pd.DataFrame({"x": [1]})
    paths = save_csv_outputs(
        tmp_path,
        benchmark_metrics=frame,
        per_image=frame,
        per_image_with_threshold=frame,
        threshold_summary=frame,
        threshold_sweep=frame,
        confusion_matrix=frame,
        false_positives=frame,
        false_negatives=frame,
        defect_breakdown=frame,
    )
    assert (tmp_path / "benchmark_metrics.csv").exists()
    assert "per_image_scores" in paths
    report = save_report(tmp_path, "# Report")
    assert report.exists()
    zip_path = create_zip_archive(tmp_path, "project")
    assert zip_path.exists()
