import pytest

np = pytest.importorskip("numpy")

from movebench.config import LAYERS, LAYER_ORDER
from movebench.layer_pipeline import LayerPipeline
from movebench.utils import make_placeholder


def _pipeline(tmp_path):
    return LayerPipeline(tmp_path, tmp_path / "chips")


def test_pipeline_order_matches_config(tmp_path):
    pipeline = _pipeline(tmp_path)
    assert pipeline.layer_order == LAYER_ORDER


def test_pipeline_dependencies_match_config(tmp_path):
    pipeline = _pipeline(tmp_path)
    deps = {task.name: task.dependencies for task in pipeline.tasks}
    for name, config in LAYERS.items():
        if config.derived_from:
            assert deps[name] == config.derived_from


def test_placeholder_generation_uses_layer_config():
    ref_shape = (4, 5)
    placeholder = make_placeholder(
        LAYERS["dynamic_world"],
        shape=ref_shape,
        transform="transform",
        crs="EPSG:4326",
        bounds=(0, 0, 1, 1),
    )

    expected_bands = len(LAYERS["dynamic_world"].bands)
    assert placeholder["array"].shape == (expected_bands, *ref_shape)
    assert placeholder["bands"] == LAYERS["dynamic_world"].bands
    assert np.isnan(placeholder["array"]).all()
