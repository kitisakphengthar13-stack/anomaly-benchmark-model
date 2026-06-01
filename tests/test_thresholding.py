import pandas as pd

from abm.config import ThresholdConfig
from abm.thresholding import apply_threshold, generate_threshold_sweep


def frame():
    return pd.DataFrame({"y_true": [0, 0, 1, 1], "y_pred": [0, 1, 1, 0], "score": [0.1, 0.4, 0.8, 0.9]})


def test_generate_threshold_sweep():
    sweep = generate_threshold_sweep(frame(), 5)
    assert list(sweep.columns) == ["threshold", "precision", "recall", "f1"]
    assert len(sweep) == 5


def test_best_f1_threshold():
    result = apply_threshold(frame(), ThresholdConfig(method="best_f1", num_thresholds=10))
    assert "threshold_y_pred" in result.per_image.columns
    assert not result.sweep.empty


def test_manual_threshold():
    result = apply_threshold(frame(), ThresholdConfig(method="manual", manual_value=0.5))
    assert result.per_image["threshold_y_pred"].tolist() == [0, 0, 1, 1]


def test_use_model_threshold():
    result = apply_threshold(frame(), ThresholdConfig(method="use_model_threshold"))
    assert result.threshold == "model_provided"
    assert result.per_image["threshold_y_pred"].tolist() == [0, 1, 1, 0]
