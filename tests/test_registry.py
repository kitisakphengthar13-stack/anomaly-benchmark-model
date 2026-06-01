import pytest

from abm.data_factory import filter_kwargs_for_signature
from abm.registry import DATASET_ALIASES, DATASET_REGISTRY, MODEL_ALIASES, MODEL_REGISTRY, canonical_name, resolve_class
from abm.utils import BenchmarkError


class Dummy:
    def __init__(self, root=None, eval_batch_size=None):
        self.root = root
        self.eval_batch_size = eval_batch_size


def test_model_alias_resolution():
    assert canonical_name("rd", MODEL_REGISTRY, MODEL_ALIASES) == "ReverseDistillation"
    assert canonical_name("patchcore", MODEL_REGISTRY, MODEL_ALIASES) == "Patchcore"


def test_dataset_alias_resolution():
    assert canonical_name("custom", DATASET_REGISTRY, DATASET_ALIASES) == "Folder"


def test_dynamic_class_path_resolution():
    cls = resolve_class(name=None, class_path="pathlib.Path", registry={}, aliases={}, kind="model")
    assert cls.__name__ == "Path"


def test_unknown_name_error_mentions_class_path():
    with pytest.raises(BenchmarkError, match="class_path"):
        canonical_name("Nope", MODEL_REGISTRY, MODEL_ALIASES)


def test_signature_filter_strict_raises():
    with pytest.raises(BenchmarkError, match="Unsupported dataset args"):
        filter_kwargs_for_signature(Dummy, {"root": "x", "image_size": 256}, strict=True)


def test_signature_filter_non_strict_skips():
    filtered = filter_kwargs_for_signature(Dummy, {"root": "x", "image_size": 256}, strict=False)
    assert filtered == {"root": "x"}
