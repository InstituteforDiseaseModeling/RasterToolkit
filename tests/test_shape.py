import numpy as np
import pytest
import re

from typing import Dict

from rastertools import ShapeView, area_sphere, centroid_area, shape_subdivide
from rastertools.shape import shapes_to_polygons, shapes_to_polygons_dict, polygon_area_km2
from pytest_init import change_test_dir  # don't remove 

from pathlib import Path


def setup_module():
    pytest.expected_name = "AFRO:DRCONGO:HAUT_KATANGA:KAMPEMBA"


@pytest.fixture()
def shape_file() -> Path:
    return "data/cod_lev02_zones_test/cod_lev02_zones_test"


@pytest.fixture()
def one_shape(shape_file) -> ShapeView:
    """Helper test property providing a sample shape view object."""
    shapes_dict = {s.name: s for s in ShapeView.from_file(shape_file)}
    return shapes_dict[pytest.expected_name]


@pytest.mark.unit
def test_shape_load_from_file():
    """Testing loading of a shape file and creating a list of shape view objects."""
    shapes = ShapeView.from_file(pytest.shape_file)
    assert len(shapes) > 0
    for shp in shapes:
        shp.validate()


@pytest.mark.unit
def test_shape_properties(one_shape):
    """Testing shape view object properties."""
    shp = one_shape
    assert isinstance(shp, ShapeView)

    # name, parts_count, areas
    assert pytest.expected_name == shp.name
    assert shp.parts_count == 2
    assert round(len(shp.areas), 0) == 2
    assert round(shp.areas[0], 4) == 729.4677

    # xy min/max
    assert isinstance(shp.xy_max, np.ndarray)
    assert isinstance(shp.xy_min, np.ndarray)
    assert round(shp.xy_max[0], 4) == 28.0105
    assert round(shp.xy_max[1], 4) == -11.5730
    assert round(shp.xy_min[0], 4) == 27.6754
    assert round(shp.xy_min[1], 4) == -11.959

    # points
    assert isinstance(shp.points, np.ndarray)
    assert shp.points.shape[0] > 2
    assert shp.points.shape[1] == 2
    assert np.array_equal(shp.points[0, :], shp.points[-1, :])

    # centroid
    assert round(shp.center[0],4) == 27.8632
    assert round(shp.center[1], 4) == -11.7542


@pytest.mark.unit
def test_shape_area_sphere(one_shape):
    """Testing the function for calculating shape sphere area."""
    parts = one_shape.shape.parts
    points: np.ndarray = one_shape.points[parts[0]:parts[1]]
    actual_area = area_sphere(points)
    assert round(actual_area, 4), 729.4677


@pytest.mark.unit
def test_shape_centroid_area(one_shape):
    shp = one_shape

    prt_list = list(shp.shape.parts) + [len(shp.points)]
    points = shp.points[prt_list[0]:prt_list[1]]
    x1, y1, a1 = centroid_area(points)

    assert round(x1, 4) == 27.8632
    assert round(y1, 4) == -11.7542
    assert round(abs(a1), 4) == 0.0603


@pytest.mark.unit
def test_shape_sub_simple(one_shape, shape_file, tmp_path):
    out_shape_stem = tmp_path.joinpath("sub_simple")
    shape_subdivide(shape_stem=shape_file, out_shape_stem=out_shape_stem)

    # Verify
    sub_shapes = [s for s in ShapeView.from_file(out_shape_stem) if s.name.startswith(pytest.expected_name)]
    names = [s.name[len(pytest.expected_name)+1:] for s in sub_shapes]
    names_ok = [re.match("^[A-Z]0{3}[0-9]$", n) is not None for n in names]
    assert all(names_ok), "Shape names must match the pattern."  # "AFRO:DRCONGO:HAUT_KATANGA:KAMPEMBA:A0001"

    expected_area = one_shape.area_km2
    actual_area = sum([s.area_km2 for s in sub_shapes])
    assert round(expected_area, 2) == round(actual_area, 2)
