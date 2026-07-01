import numpy as np
import pytest
import re

from rastertoolkit.shape import ShapeView, area_sphere, centroid_area, shape_subdivide

from pathlib import Path
from shapefile import Reader, Writer


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """
    Ensure the correct working directory is set.
    """
    monkeypatch.chdir(request.fspath.dirname)


def setup_module():
    pytest.expected_name = "AFRO:DRCONGO:HAUT_KATANGA:KAMPEMBA"


@pytest.fixture()
def shape_file() -> Path:
    return Path("data/cod_lev02_zones_test/cod_lev02_zones_test")


@pytest.fixture()
def one_shape(shape_file) -> ShapeView:
    """
    Helper test property providing a sample shape view object.
    """
    shapes_dict = {s.name: s for s in ShapeView.from_file(shape_file)}
    return shapes_dict[pytest.expected_name]


def test_shape_load_from_file(shape_file):
    """
    Testing loading of a shape file and creating a list of shape view objects.
    """
    shapes = ShapeView.from_file(shape_file)
    assert len(shapes) > 0
    for shp in shapes:
        shp.validate()


def test_shape_load_from_file_filter(shape_file):
    """
    Testing loading of a single shape from a file of many shapes.
    """
    shapes = ShapeView.from_file(shape_file, attr_filter=pytest.expected_name)
    assert len(shapes) == 1
    for shp in shapes:
        shp.validate()


def test_shape_properties(one_shape):
    """
    Testing shape view object properties.
    """
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
    assert round(shp.center[0], 4) == 27.8632
    assert round(shp.center[1], 4) == -11.7542


def test_shape_area_sphere(one_shape):
    """
    Testing the function for calculating shape sphere area.
    """
    parts = one_shape.shape.parts
    points: np.ndarray = one_shape.points[parts[0]:parts[1]]
    actual_area = area_sphere(points)
    assert round(actual_area, 4) == 729.4677


def test_shape_centroid_area(one_shape):
    shp = one_shape

    prt_list = list(shp.shape.parts) + [len(shp.points)]
    points = shp.points[prt_list[0]:prt_list[1]]
    x1, y1, a1 = centroid_area(points)

    assert round(x1, 4) == 27.8632
    assert round(y1, 4) == -11.7542
    assert round(abs(a1), 4) == 0.0603


# Subdivision Tests


def test_shape_sub_default(one_shape, shape_file):
    run_shape_sub_test(one_shape, shape_file)


def test_shape_sub_shp(one_shape, shape_file):
    run_shape_sub_test(one_shape, shape_file.with_suffix(".shp"))


def test_shape_sub_temp_dir(one_shape, shape_file, tmp_path):
    run_shape_sub_test(one_shape, shape_file, tmp_path)


def test_shape_sub_400km(one_shape, shape_file):
    run_shape_sub_test(one_shape, shape_file, target_area=400)


def test_shape_sub_120pt(one_shape, shape_file):
    run_shape_sub_test(one_shape, shape_file, points_per_box=120)


def test_shape_sub_seed(one_shape, shape_file):
    run_shape_sub_test(one_shape, shape_file, random_seed=10)


def run_shape_sub_test(one_shape, shape_file, tmp_path=None, target_area=None, points_per_box=None, random_seed=None):
    out_shape_stem = shape_subdivide(shape_stem=shape_file,
                                     out_dir=tmp_path,
                                     box_target_area_km2=target_area,
                                     points_per_box=points_per_box,
                                     random_seed=random_seed)
    # Verify
    assert str(out_shape_stem).endswith(f"_{target_area or 100}km"), "Default name must end with target area."
    sub_shapes = [s for s in ShapeView.from_file(out_shape_stem) if s.name.startswith(pytest.expected_name)]
    names = [s.name[len(pytest.expected_name) + 1:] for s in sub_shapes]
    names_ok = [re.match("^[A-Z]0{3}[0-9]$", n) is not None for n in names]
    assert all(names_ok), "Shape names must match the pattern."  # "AFRO:DRCONGO:HAUT_KATANGA:KAMPEMBA:A0001"

    expected_area = one_shape.area_km2
    actual_area = sum([s.area_km2 for s in sub_shapes])
    assert round(expected_area, 1) == round(actual_area, 1)


def test_shape_sub_extra_field(shape_file, tmp_path):
    """
    Regression test: extra fields beyond DOTNAME must survive subdivision.

    pyshp 3.0 returns Field namedtuples; wrapping them with tuple() strips
    the .size attribute that pyshp needs when writing the DBF header.
    """
    # Create a copy of the test shapefile with an extra field
    extra_shape_stem = tmp_path / "extra_field_test"
    with Reader(str(shape_file)) as sf_in:
        with Writer(str(extra_shape_stem)) as sf_out:
            # Copy all existing fields
            for field in sf_in.fields[1:]:  # skip DeletionFlag
                sf_out.field(*field)
            # Add an extra numeric field
            sf_out.field("EXTRA", "N", 10, 0)
            # Copy shapes and records, appending a value for the extra field
            for i, shaperec in enumerate(sf_in.iterShapeRecords()):
                sf_out.shape(shaperec.shape)
                sf_out.record(*shaperec.record, i + 1)

    # Subdivide the shapefile with the extra field
    out_shape_stem = shape_subdivide(
        shape_stem=extra_shape_stem,
        out_dir=tmp_path,
        box_target_area_km2=100,
    )

    # Verify the extra field is present in the output
    with Reader(str(out_shape_stem)) as sf_out:
        field_names = [f[0] for f in sf_out.fields[1:]]  # skip DeletionFlag
        assert "EXTRA" in field_names, "Extra field must be preserved in output"

        # Verify that records have values for the extra field
        for rec in sf_out.records():
            assert rec["EXTRA"] is not None
