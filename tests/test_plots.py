import matplotlib
import pandas as pd

from abm.config import PlotsConfig
from abm.plots import save_plots
from abm.thresholding import generate_threshold_sweep


matplotlib.use("Agg")


def test_human_readable_report_plots_are_generated(tmp_path):
    df = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "threshold_y_pred": [0, 1, 1, 0],
            "score": [0.1, 0.7, 0.8, 0.2],
        }
    )
    sweep = generate_threshold_sweep(df, 5)

    paths = save_plots(df, sweep, tmp_path, PlotsConfig(), threshold=0.5)

    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "score_distribution.png").exists()
    assert (tmp_path / "per_image_score.png").exists()
    assert (tmp_path / "threshold_sweep.png").exists()
    assert not (tmp_path / "confidence_vs_score.png").exists()
    assert set(paths) == {"confusion_matrix", "score_distribution", "per_image_score", "threshold_sweep"}
