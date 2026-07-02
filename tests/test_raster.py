import json
import os
import pytest

import numpy as np

from rastertoolkit import raster_clip, raster_clip_weighted


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """
    Ensure the correct working directory is set.
    """
    monkeypatch.chdir(request.fspath.dirname)


def setup_function() -> None:
    pytest.shape_file = "data/cod_lev02_zones_test/cod_lev02_zones_test"
    pytest.raster_file = "data/cod_2012_1km_aggregated_unadj_test.tif"
    pytest.vacc_raster_file = "data/IHME_MCV1_2012_MEAN_test.tif"


def test_raster_clip():
    """
    Testing raster_clip with the default stats function (sum).
    """
    actual_pop = raster_clip(pytest.raster_file, pytest.shape_file)
    with open(os.path.join("expected", "clipped_pop_sum.json")) as fid01:
        expected_pop = json.load(fid01)

    assert expected_pop == actual_pop


def test_raster_clip_stat_fn():
    """
    Testing raster_clip with a provided stats function.
    """
    actual_mean_pop = raster_clip(pytest.raster_file, pytest.shape_file, summary_func=np.mean)
    with open(os.path.join("expected", "clipped_pop_sum.json")) as fid01:
        expected_sum_pop = json.load(fid01)
    with open(os.path.join("expected", "clipped_pop_mean.json")) as fid01:
        expected_mean_pop = json.load(fid01)

    for k in actual_mean_pop:
        assert round(expected_mean_pop[k], 4) == round(actual_mean_pop[k], 4)
        assert expected_sum_pop[k] >= int(actual_mean_pop[k])


def test_raster_clip_weighted():
    actual_weighted = raster_clip_weighted(pytest.raster_file, pytest.vacc_raster_file, pytest.shape_file)
    with open(os.path.join("expected", "clipped_pop_weighted_sum.json")) as fid01:
        expected_weighted = json.load(fid01)

    assert all([not np.isnan(v["pop"]) for v in actual_weighted.values()]), "One or more pop values are NaN."
    assert all([not np.isnan(v["val"]) for v in actual_weighted.values()]), "One or more weighted vacc value is NaN."
    assert expected_weighted == actual_weighted
