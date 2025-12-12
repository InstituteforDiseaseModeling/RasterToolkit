import numpy as np
import pytest

from pathlib import Path
from typing import Dict

from rastertoolkit import raster_clip, raster_clip_weighted, utils


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """Ensure the correct working directory is set."""
    monkeypatch.chdir(request.fspath.dirname)


def setup_function() -> None:
    pytest.shape_file = "data/cod_lev02_zones_test/cod_lev02_zones_test"
    pytest.raster_file = "data/cod_2012_1km_aggregated_unadj_test.tif"
    pytest.vacc_raster_file = "data/IHME_MCV1_2012_MEAN_test.tif"


@pytest.mark.unit
def test_raster_clip():
    """Testing raster_clip with the default stats function (sum)."""
    actual_pop: Dict = raster_clip(pytest.raster_file, pytest.shape_file)
    expected_pop: Dict = utils.read_json(Path("expected").joinpath("clipped_pop_sum.json"))
    assert expected_pop == actual_pop


@pytest.mark.unit
def test_raster_clip_stat_fn():
    """Testing raster_clip with a provided stats function."""
    actual_mean_pop: Dict = raster_clip(pytest.raster_file, pytest.shape_file, summary_func=np.mean)
    expected_sum_pop: Dict = utils.read_json(Path("expected").joinpath("clipped_pop_sum.json"))
    expected_mean_pop: Dict = utils.read_json(Path("expected").joinpath("clipped_pop_mean.json"))

    for k in actual_mean_pop:
        assert round(expected_mean_pop[k], 4) == round(actual_mean_pop[k], 4)
        assert expected_sum_pop[k] >= int(actual_mean_pop[k])


@pytest.mark.unit
def test_raster_clip_weighted():
    actual_weighted: Dict = raster_clip_weighted(pytest.raster_file, pytest.vacc_raster_file, pytest.shape_file)
    expected_weighted: Dict = utils.read_json(Path("expected").joinpath("clipped_pop_weighted_sum.json"))
    assert all([not np.isnan(v["pop"]) for v in actual_weighted.values()]), "One or more pop values are NaN."
    assert all([not np.isnan(v["val"]) for v in actual_weighted.values()]), "One or more weighted vacc value is NaN."
    assert expected_weighted == actual_weighted


# TODO: add unittests for helper functions form the raster module
