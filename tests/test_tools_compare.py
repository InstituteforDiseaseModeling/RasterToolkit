import numpy as np
import pytest
import sys

from shapely.geometry import Polygon
from pyproj import Geod

from rastertoolkit.shape import ShapeView, area_sphere, centroid_area


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """Ensure the correct working directory is set."""
    monkeypatch.chdir(request.fspath.dirname)


def setup_function() -> None:
    pytest.shape_file = "data/cod_lev02_zones_test/cod_lev02_zones_test"
    pytest.expected_name = "AFRO:DRCONGO:HAUT_KATANGA:KAMPEMBA"


# Sphere area function vs. pyproj
sphere_area_diff_perc = 0.005


@pytest.mark.unit
@pytest.mark.skipif('pyproj' not in sys.modules, reason="requires the 'pyproj' library")
def test_area_sphere_vs_pyproj_simple():
    """Compare sphere area diff for simple cases"""
    all_diff_perc = []
    all_cases = [
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        [[70.0, 30.0], [70.0, 31.0], [71.0, 31.0], [71.0, 30.0], [70.0, 30.0]]]

    for case in all_cases:
        points = np.array(case)
        diff_km2, diff_perc = calc_sphere_area_diff(points)
        assert diff_perc < sphere_area_diff_perc  # less than 1 km2 (tentative, current state)
        all_diff_perc.append(diff_perc)

    assert np.mean(all_diff_perc) < sphere_area_diff_perc * 0.6


@pytest.mark.unit
@pytest.mark.skipif('pyproj' not in sys.modules, reason="requires the 'pyproj' library")
def test_area_sphere_vs_pyproj_many():
    """Compare sphere area diff for shapes in the test shape file."""
    shapes = ShapeView.from_file(pytest.shape_file)
    all_diff_perc = []
    for shp in shapes:
        prt_list = list(shp.shape.parts) + [len(shp.points)]
        for i in range(len(prt_list) - 1):
            # Actual sphere ares
            points = shp.points[prt_list[i]:prt_list[i + 1]]
            diff_km2, diff_perc = calc_sphere_area_diff(points)
            assert diff_perc < sphere_area_diff_perc  # less than 1 km2 (tentative, current state)
            all_diff_perc.append(diff_perc)

    assert np.mean(all_diff_perc) < sphere_area_diff_perc * 0.6  # less than 1% (tentative, current state)


def calc_sphere_area_diff(points: np.ndarray):
    # Actual sphere ares
    actual_km2 = area_sphere(points)

    # Expected sphere area using pyproj
    # https://pyproj4.github.io/pyproj/stable/api/geod.html#pyproj.Geod.polygon_area_perimeter
    geo_area_m2, _ = Geod(ellps="WGS84").geometry_area_perimeter(Polygon(points))
    expected_km2 = -1 * geo_area_m2 / 10 ** 6  # geodesic area (km^2), "-1" matching orientation

    # Compare diff
    # TODO: determine tolerance
    diff_km2 = abs(round(actual_km2, 2) - round(expected_km2, 2))
    diff_perc = diff_km2 / expected_km2
    return diff_km2, diff_perc


# Centroid function vs. Shapely


@pytest.mark.unit
@pytest.mark.skipif('shapely' not in sys.modules, reason="requires the 'Shapely' library")
def test_centroid_area_all_shapes():
    """Testing the function for calculating shape centroid."""
    shapes = ShapeView.from_file(pytest.shape_file)

    for shp in shapes:
        prt_list = list(shp.shape.parts) + [len(shp.points)]
        for i in range(len(prt_list) - 1):
            # Skip a known edge case (see "test_centroid_area_edge_case")
            if shp.name == "AFRO:DRCONGO:HAUT_KATANGA:KAMPEMBA" and i == 1:
                continue

            points = shp.points[prt_list[i]:prt_list[i + 1]]
            validate_centroid(points, places=4)


@pytest.mark.unit
@pytest.mark.skipif('shapely' not in sys.modules, reason="requires the 'Shapely' library")
def test_centroid_area_edge_case():
    points = np.array([
        [27.67629337, - 11.57355617],
        [27.67629332, - 11.57355617],
        [27.6754073, -11.57303985],
        [27.67629337, - 11.57355617]
    ])

    validate_centroid(points, places=2)  # fails for places=3 or higher


def validate_centroid(points, places=4):
    """Compare centroid coordinates and area with shapley."""
    # actual centroid
    x1, y1, a1 = centroid_area(points)

    # expected centroid (from Shapely)
    p = Polygon(points)
    c = p.centroid
    x2, y2, a2 = c.xy[0][0], c.xy[1][0], p.area

    assert round(x1, places) == round(x2, places)
    assert round(y1, places) == round(y2, places)
    assert round(abs(a1), places) == round(a2, places)
