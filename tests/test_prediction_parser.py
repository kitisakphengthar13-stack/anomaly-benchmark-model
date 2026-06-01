from pathlib import Path

import numpy as np
import pytest

from abm.config import LabelFromPathConfig
from abm.prediction_parser import parse_predictions


class ObjPrediction:
    def __init__(self):
        self.image_path = ["D:/data/good/a.png", "D:/data/bad/crack/b.png"]
        self.gt_label = [0, 1]
        self.pred_label = [0, 0]
        self.pred_score = [[0.1], [0.8]]


def test_parse_dict_prediction():
    df = parse_predictions(
        [{"image_path": "D:/data/good/a.png", "gt_label": "good", "pred_label": "bad", "pred_score": 0.7}],
        model_name="m",
        dataset_name="d",
        category=None,
    )
    assert df.loc[0, "y_true"] == 0
    assert df.loc[0, "y_pred"] == 1
    assert df.loc[0, "score"] == pytest.approx(0.7)


def test_parse_batched_object_prediction():
    df = parse_predictions(ObjPrediction(), model_name="m", dataset_name="d", category="c")
    assert len(df) == 2
    assert df.loc[1, "y_true"] == 1
    assert df.loc[1, "score"] == pytest.approx(0.8)


def test_parse_numpy_arrays():
    pred = {
        "image_path": np.array(["D:/data/good/a.png", "D:/data/bad/b.png"]),
        "gt_label": np.array([0, 1]),
        "pred_label": np.array([0, 1]),
        "pred_score": np.array([0.2, 0.9]),
    }
    df = parse_predictions(pred, model_name="m", dataset_name="d", category=None)
    assert df["score"].tolist() == [0.2, 0.9]


def test_parse_torch_tensors_if_available():
    torch = pytest.importorskip("torch")
    pred = {
        "image_path": ["D:/data/good/a.png"],
        "gt_label": torch.tensor([0]),
        "pred_label": torch.tensor([1]),
        "pred_score": torch.tensor([[0.4]]),
    }
    df = parse_predictions(pred, model_name="m", dataset_name="d", category=None)
    assert df.loc[0, "score"] == pytest.approx(0.4)


def test_missing_label_fallback_from_path():
    df = parse_predictions(
        [{"image_path": Path("D:/data/anomaly/crack/a.png"), "pred_label": 1, "pred_score": 0.9}],
        model_name="m",
        dataset_name="d",
        category=None,
        label_from_path=LabelFromPathConfig(),
    )
    assert df.loc[0, "y_true"] == 1
    assert df.loc[0, "group"] == "bad"
