"""
Functions for spatial processing of raster TIFF files.
"""

import matplotlib.path as plt
import numpy as np
import shapefile

from PIL import Image
from PIL.TiffTags import TAGS
from scipy import interpolate
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Callable
from rastertools.shape import ShapeView

Image.MAX_IMAGE_PIXELS = 933120000  # prevents the DecompressionBombWarning. Default limit is 89478485 pixels


def raster_clip(raster_file: Union[str, Path],
                shape_stem: Union[str, Path],
                shape_attr: str = "DOTNAME",
                summary_func: Callable = None,
                include_latlon: bool = False) -> Dict[str, Union[float, int]]:
    """
    Extract data from a raster based on shapes.
    :param raster_file: Local path to a raster file.
    :param shape_stem: Local path stem referencing a set of shape files.
    :param shape_attr: The shape attribute name to be use as output dictionary key.
    :param summary_func: Aggregation func to be used for summarizing clipped data for each shape.
    :param include_latlon: The flag to include lat/lon in the dict entry.
    :return: Dictionary with dot names as keys and calculates aggregations as values.
    """
    assert Path(raster_file).is_file(), "Raster file not found."

    # Load data, init sparce matrix
    shapes = ShapeView.from_file(shape_stem, shape_attr)
    raster = Image.open(raster_file)
    sparce_data = init_sparce_matrix(raster)

    # Output dictionary
    data_dict = dict()

    # Iterate over shapes in shapefile
    for k1, shp in enumerate(shapes):
        # Null shape; error in shapefile
        shp.validate()

        # Subset population data matrix for clipping
        data_clip = subset_matrix_for_clipping(shp, sparce_data)

        if data_clip.shape[0] == 0:
            data_dict[shp.name] = summary_entry(None, {"pop": 0}, include_latlon)
            print_status(shp, data_dict, k1, len(shapes))
            continue

        # Pop values
        value = data_clip[is_interior(shp, data_clip), 2]

        # Entry dictionary
        summary_func = summary_func or default_summary_func
        entry = {"pop": summary_func(value)}

        # Set entry and print status
        data_dict[shp.name] = summary_entry(shp, entry, include_latlon)
        print_status(shp, data_dict, k1, len(shapes))

    return data_dict


def raster_clip_weighted(raster_weight: Union[str, Path],
                         raster_value: Union[str, Path],
                         shape_stem: Union[str, Path],
                         shape_attr: str = "DOTNAME",
                         weight_summary_func: Callable = None,
                         include_latlon: bool = False) -> Dict[str, Union[float, int]]:
    """
    Extract data from a raster based on shapes.
    :param raster_weight: Local path to a raster file used for weights.
    :param raster_value: Local path to a raster file used for values.
    :param shape_stem: Local path stem referencing a set of shape files.
    :param shape_attr: The shape attribute name to be use as output dictionary key.
    :param weight_summary_func: Aggregation func to be used for summarizing clipped data for each shape.
    :param include_latlon: The flag to include lat/lon in the dict entry.
    :return: Dictionary with dot names as keys and calculates aggregations as values.
    """
    assert Path(raster_weight).is_file(), "Population raster file not found."
    assert Path(raster_value).is_file(), "Values raster file not found."

    # Load data shape and rasters
    shapes = ShapeView.from_file(shape_stem, shape_attr)
    raster_weights = Image.open(raster_weight)
    raster_values = Image.open(raster_value)

    # Init sparce matrices
    sparce_pop = init_sparce_matrix(raster_weights)
    sparce_val = init_sparce_matrix(raster_values)

    # Output dictionary
    data_dict = dict()

    # Iterate over shapes in shapefile
    for k1, shp in enumerate(shapes):
        # Null shape; error in shapefile
        shp.validate()

        # Subset matrices for clipping
        pop_clip = subset_matrix_for_clipping(shape=shp, sparce_data=sparce_pop)
        val_clip = subset_matrix_for_clipping(shape=shp, sparce_data=sparce_val, pad=1)

        # Track booleans (indicates if lat/long is interior)
        data_bool = is_interior(shp, pop_clip)

        # Interpolate at population data
        final_val = interpolate_at_weight_data(shp, pop_clip, val_clip, data_bool)

        # Pop values
        values = pop_clip[data_bool, 2]

        # Entry dictionary
        weight_summary_func = weight_summary_func or default_summary_func
        entry = {"pop": weight_summary_func(values), "val": final_val}

        # Set entry and print status
        data_dict[shp.name] = summary_entry(shp, entry, include_latlon)
        print_status(shp, data_dict, k1, len(shapes))

    return data_dict


def default_summary_func(v: np.ndarray) -> int:
    return int(np.round(np.sum(v), 0))


def get_tiff_tags(raster: Image) -> Dict[str, Any]:
    """
    Read tags from a TIFF file
    https://stackoverflow.com/questions/46477712/reading-tiff-image-metadata-in-python
    :param raster: TIFF object
    :return: Dictionary of tag names and values.
    """
    return {TAGS[t]: raster.tag[t] for t in dict(raster.tag)}


def extract_xy_info_from_raster(raster: Image) -> Tuple[float, float, float, float]:
    # Extract data from raster
    tags = get_tiff_tags(raster)
    point = tags["ModelTiepointTag"]
    scale = tags["ModelPixelScaleTag"]
    x0, y0 = point[3], point[4]
    dx, dy = scale[0], -scale[1]

    # Make sure values are in range
    assert -180 < x0 < 180, "Tie point x coordinate (longitude) have invalid range."
    assert -85 < y0 < 85, "Tie point y coordinate (latitude) have invalid range."
    assert 0 < dx < 1, "Pixel dx scale has invalid range."
    assert -1 < dy < 0, "Pixel dy scale has invalid range."

    return x0, y0, dx, dy


def init_sparce_matrix(raster: Image) -> np.ndarray:
    # Extract data from raster
    x0, y0, dx, dy = extract_xy_info_from_raster(raster)

    dat_mat = np.array(raster)
    xy_ints = np.argwhere(dat_mat > 0)
    sparce_data = np.zeros((xy_ints.shape[0], 3), dtype=float)

    # Construct sparce matrix of (long, lat, data)
    sparce_data[:, 0] = x0 + dx * xy_ints[:, 1] + dx / 2.0
    sparce_data[:, 1] = y0 + dy * xy_ints[:, 0] + dy / 2.0
    sparce_data[:, 2] = dat_mat[xy_ints[:, 0], xy_ints[:, 1]]

    return sparce_data


def subset_matrix_for_clipping(shape: ShapeView, sparce_data: np.ndarray, pad: int = 0) -> np.ndarray:
    clip_bool1 = np.logical_and(sparce_data[:, 0] > shape.xy_min[0] - pad, sparce_data[:, 1] > shape.xy_min[1] - pad)
    clip_bool2 = np.logical_and(sparce_data[:, 0] < shape.xy_max[0] + pad, sparce_data[:, 1] < shape.xy_max[1] + pad)
    data_clip = sparce_data[np.logical_and(clip_bool1, clip_bool2), :]

    return data_clip


def summary_entry(shape: ShapeView, entry: Union[Dict, float, int], include_latlon: bool) -> Union[Dict, float, int]:
    if include_latlon:
        assert isinstance(entry, dict) and len(entry) > 0, "Invalid entry."
        lon = shape.center[0] if shape else np.nan
        lat = shape.center[1] if shape else np.nan
        final_entry = {"lat": lat, "lon": lon}
        final_entry.update(entry)
    else:
        if isinstance(entry, dict) and len(entry) == 1:
            final_entry = list(entry.values())[0]
        else:
            final_entry = entry

    return final_entry


def is_interior(shape: ShapeView, data_clip: np.ndarray) -> bool:
    # Track booleans (indicates if lat/long is interior)
    data_bool = np.zeros(data_clip.shape[0], dtype=bool)

    # Iterate over parts of shapefile
    for path_shp, area_prt in zip(shape.paths, shape.areas):
        # Union of positive areas; intersection with negative areas
        if area_prt > 0:
            data_bool = np.logical_or(data_bool, path_shp.contains_points(data_clip[:, :2]))
        else:
            data_bool = np.logical_and(data_bool, np.logical_not(path_shp.contains_points(data_clip[:, :2])))

    return data_bool


def print_status(shape: ShapeView, data_dict: Dict, k1: int, shape_count: int) -> None:
    print(k1 + 1, 'of', shape_count, shape.name, shape.center, data_dict[shape.name])


def interpolate_at_weight_data(shape: ShapeView,
                               weight_clip: np.ndarray,
                               value_clip: np.ndarray,
                               data_bool: bool) -> float:
    # Calculate population weighted value
    weight = np.sum(weight_clip[data_bool, 2])

    # Prep interpolate coordinates and value arguments
    value_args = [value_clip[:, 0:2], value_clip[:, 2]]

    if weight > 0:
        # Interpolate at weight, assign -1 for problems
        val_est = interpolate.griddata(*value_args, weight_clip[:, 0:2], fill_value=-1)
        if -1 in val_est:
            err_dex = (val_est == -1)
            # Use the nearest value for problems
            val_rev = interpolate.griddata(*value_args, weight_clip[err_dex, 0:2], method='nearest')
            val_est[err_dex] = val_rev
        # Use population to weight values
        final_val = np.sum(weight_clip[data_bool, 2] * val_est[data_bool]) / weight
    else:
        # No population data, interpolate at boundary, assign -1 for problems
        val_est = interpolate.griddata(*value_args, shape.points[:, 0:2], fill_value=-1)
        if -1 in val_est:
            err_dex = (val_est == -1)
            # Use the nearest value for problems
            val_rev = interpolate.griddata(*value_args, shape.points[err_dex, 0:2], method='nearest')
            val_est[err_dex] = val_rev

        # Average values at shape perimeter
        final_val = np.mean(val_est)

    return final_val
