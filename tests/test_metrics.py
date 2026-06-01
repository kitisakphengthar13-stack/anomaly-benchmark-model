import pandas as pd

from abm.metrics import (
    confusion_matrix_table,
    defect_type_breakdown,
    false_negative_table,
    false_positive_table,
    scalar_metrics,
)


def test_metrics_tables():
    df = pd.DataFrame(
        {
            "image_path": ["a", "b", "c", "d"],
            "defect_type": ["good", "good", "crack", "dent"],
            "y_true": [0, 0, 1, 1],
            "threshold_y_pred": [0, 1, 1, 0],
            "score": [0.1, 0.7, 0.8, 0.2],
        }
    )
    cm = confusion_matrix_table(df).iloc[0].to_dict()
    assert cm == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert len(false_positive_table(df)) == 1
    assert len(false_negative_table(df)) == 1
    assert scalar_metrics(df)["f1"] == 0.5
    breakdown = defect_type_breakdown(df)
    assert breakdown.iloc[0]["defect_type"] == "dent"
