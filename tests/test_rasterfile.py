import numpy as np
import pytest
from pathlib import Path
from tifffile import TiffFile, TiffWriter

from rastertoolkit.raster import get_tiff_tags, init_sparse_matrix

# GeoTIFF tags for a 10x10 grid: origin lon=10, lat=5, resolution=0.1 degrees.
# GDAL_NODATA (42113) is set to -1 so extract_xy_info_from_raster's nodata check passes.
_TIEPOINT = (33922, 12, 6, [0.0, 0.0, 0.0, 10.0, 5.0, 0.0], False)
_SCALE    = (33550, 12, 3, [0.1, 0.1, 0.0], False)
_NODATA   = (42113,  2, 3, b"-1", False)
_GEO_TAGS = [_TIEPOINT, _SCALE, _NODATA]


@pytest.fixture
def geotiff_pages_bands(tmp_path):
    """3-page GeoTIFF with 2D / 1-band / 3-band pages and valid GeoTIFF coordinate tags."""
    page0 = np.zeros((10, 10), dtype=float)
    page0[2, 3] = 100.0
    page0[5, 7] = 200.0

    # tifffile squeezes (10, 10, 1) back to (10, 10) on read; stored as single-band.
    page1 = np.zeros((10, 10, 1), dtype=float)
    page1[2, 3, 0] = 50.0
    page1[5, 7, 0] = 150.0

    page2 = np.zeros((10, 10, 3), dtype=float)
    page2[2, 3, 0] = 10.0
    page2[2, 3, 1] = 20.0
    page2[2, 3, 2] = 30.0
    page2[5, 7, 0] = 40.0
    page2[5, 7, 1] = 50.0
    page2[5, 7, 2] = 60.0

    fp = tmp_path / "test_pages_bands.tif"
    with TiffWriter(str(fp)) as tiff:
        tiff.write(page0, extratags=_GEO_TAGS)
        tiff.write(page1, extratags=_GEO_TAGS)
        tiff.write(page2, photometric="rgb", extratags=_GEO_TAGS)
    return fp



# ---- get_tiff_tags ----

def test_get_tiff_tags_returns_dict(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    tags = get_tiff_tags(raster.pages[0])
    assert isinstance(tags, dict)


def test_get_tiff_tags_contains_geotiff_keys(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    tags = get_tiff_tags(raster.pages[0])
    assert "ModelTiepointTag" in tags
    assert "ModelPixelScaleTag" in tags


# ---- init_sparse_matrix: page 0 (2D) ----

def test_init_sparse_matrix_page0_shape(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    matrix = init_sparse_matrix(raster.pages[0], 0)
    assert matrix.ndim == 2 and matrix.shape[1] == 3


def test_init_sparse_matrix_page0_values(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    matrix = init_sparse_matrix(raster.pages[0], 0)
    assert set(matrix[:, 2]) == {100.0, 200.0}


# ---- init_sparse_matrix: page 1 (single-band, read as 2D) ----

def test_init_sparse_matrix_page1_band0_values(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    matrix = init_sparse_matrix(raster.pages[1], 0)
    assert set(matrix[:, 2]) == {50.0, 150.0}


# ---- init_sparse_matrix: page 2 (3D, 3 bands) ----

def test_init_sparse_matrix_page2_band0_values(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    matrix = init_sparse_matrix(raster.pages[2], 0)
    assert set(matrix[:, 2]) == {10.0, 40.0}


def test_init_sparse_matrix_page2_band1_values(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    matrix = init_sparse_matrix(raster.pages[2], 1)
    assert set(matrix[:, 2]) == {20.0, 50.0}


def test_init_sparse_matrix_page2_band2_values(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    matrix = init_sparse_matrix(raster.pages[2], 2)
    assert set(matrix[:, 2]) == {30.0, 60.0}


def test_bands_have_distinct_values(geotiff_pages_bands):
    """Each band on a multi-band page produces an independent set of values."""
    raster = TiffFile(geotiff_pages_bands)
    page = raster.pages[2]
    vals = [set(init_sparse_matrix(page, b)[:, 2]) for b in range(3)]
    assert vals[0].isdisjoint(vals[1])
    assert vals[1].isdisjoint(vals[2])
    assert vals[0].isdisjoint(vals[2])


def test_different_pages_give_different_values(geotiff_pages_bands):
    """Selecting different pages returns different values."""
    raster = TiffFile(geotiff_pages_bands)
    vals_p0 = set(init_sparse_matrix(raster.pages[0], 0)[:, 2])
    vals_p1 = set(init_sparse_matrix(raster.pages[1], 0)[:, 2])
    assert vals_p0 != vals_p1


# ---- init_sparse_matrix: error cases ----

def test_invalid_band_on_2d_page_raises(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    with pytest.raises(ValueError, match="Invalid raster band"):
        init_sparse_matrix(raster.pages[0], 1)


def test_band_out_of_bounds_on_3d_page_raises(geotiff_pages_bands):
    raster = TiffFile(geotiff_pages_bands)
    with pytest.raises(ValueError, match="Invalid raster band"):
        init_sparse_matrix(raster.pages[2], 3)


# ---- ModelTransformationTag ----

@pytest.fixture
def geotiff_transformation_tag(tmp_path):
    """Single-page GeoTIFF tagged with ModelTransformationTag instead of tiepoint+scale."""
    # 4x4 affine matrix (row-major): dx=0.1, dy=-0.1, origin lon=10, lat=5
    transform = [0.1, 0.0, 0.0, 10.0,
                 0.0, -0.1, 0.0, 5.0,
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0]
    transform_tag = (34264, 12, 16, transform, False)
    nodata_tag    = (42113,  2,  3, b"-1", False)

    data = np.zeros((10, 10), dtype=float)
    data[2, 3] = 100.0
    data[5, 7] = 200.0

    fp = tmp_path / "test_transform_tag.tif"
    with TiffWriter(str(fp)) as tiff:
        tiff.write(data, extratags=[transform_tag, nodata_tag])
    return fp


def test_get_tiff_tags_contains_transformation_tag(geotiff_transformation_tag):
    raster = TiffFile(geotiff_transformation_tag)
    tags = get_tiff_tags(raster.pages[0])
    assert "ModelTransformationTag" in tags
    assert "ModelTiepointTag" not in tags


def test_init_sparse_matrix_with_transformation_tag(geotiff_transformation_tag):
    """ModelTransformationTag yields the same (lon, lat, value) result as tiepoint+scale."""
    raster = TiffFile(geotiff_transformation_tag)
    matrix = init_sparse_matrix(raster.pages[0], 0)
    assert set(matrix[:, 2]) == {100.0, 200.0}
    assert all(-180 < lon < 180 for lon in matrix[:, 0])
    assert all(-85  < lat < 85  for lat in matrix[:, 1])
