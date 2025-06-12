"""
Functions for spatial processing of raster TIFF files.
"""

import numpy as np
import os

from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from PIL.TiffTags import TAGS
from scipy import interpolate
from pathlib import Path
from typing import Any, Union, Callable
from rastertoolkit.shape import ShapeView


def raster_clip(
    raster_file: Union[str, Path],
    shape_stem: Union[str, Path],
    shape_attr: str = "DOTNAME",
    attr_filter: Union[str, None] = None,
    summary_func: Callable = None,
    include_latlon: bool = False,
    quiet: bool = False,
) -> dict[str, Union[float, int]]:
    """
    Extracts data from a raster based on shapes.

    Args:
        raster_file (str): Local path to a raster file.
        shape_stem (str): Local path stem referencing a set of shape files.
        shape_attr (str): The shape attribute name to be used as the output dictionary key.
        summary_func (Callable): Aggregation function to be used for summarizing clipped data for each shape.
        include_latlon (bool, optional): Flag to include lat/lon in the dictionary entry. Defaults to False.
        quiet (bool, optional): Flag to control whether status messages are printed. Defaults to False.

    Returns:
        dict: A dictionary with dot names as keys and calculated aggregations as values.
    """
    assert Path(raster_file).is_file(), "Raster file not found."

    print("Loading data...")

    # Load data, init sparce matrix
    shapes = ShapeView.from_file(shape_stem, shape_attr, attr_filter)
    raster = Image.open(raster_file)
    sparce_data = init_sparce_matrix(raster)

    # Output dictionary
    data_dict = dict()
    shape_len = len(shapes)
    print("Clipping:")

    fts = {}
    # Init the futures executor
    executor = ThreadPoolExecutor(max_workers=(os.cpu_count() - 1))

    # Iterate over shapes in shapefile
    for k1, shp in enumerate(shapes):
        fts[k1] = executor.submit(
            raster_clip_single,
            shp=shp,
            sparce_data=sparce_data,
            k1=k1,
            shape_len=shape_len,
            summary_func=summary_func,
            include_latlon=include_latlon,
            quiet=quiet,
        )

    data_dict = {}
    for k1, ft in fts.items():
        data_dict.update(ft.result())

    executor.shutdown(wait=True)

    return data_dict


def raster_clip_single(
    shp, sparce_data, k1, shape_len, summary_func, include_latlon, quiet
):
    """
    Extracts data from a raster based on shapes.

    Args:
        shp (ShapeView): Shape object.
        sparce_data (np.ndarray): Sparce matrix of raster data.
        k1 (int): Index of the shape.
        shape_len (int): Total number of shapes.
        summary_func (Callable): Aggregation function to be used for summarizing clipped data for each shape.
        include_latlon (bool): Flag to include lat/lon in the dictionary entry.
        quiet (bool): Flag to control whether status messages are printed.

    Returns:
        dict: A dictionary with dot names as keys and calculated aggregations as values.
    """
    data_dict = {}
    show_status = not quiet or k1 % 1000 == 0 or k1 in [0, shape_len - 1]
    # Null shape; error in shapefile
    shp.validate()

    # Subset population data matrix for clipping
    data_clip = subset_matrix_for_clipping(shp, sparce_data)

    if data_clip.shape[0] == 0:
        data_dict[shp.name] = summary_entry(shp, {"pop": 0}, include_latlon)
        if show_status:
            print_status(shp, data_dict, k1, shape_len)
        return data_dict

    # Pop values
    value = data_clip[is_interior(shp, data_clip), 2]

    # Entry dictionary
    summary_func = summary_func or default_summary_func
    entry = {"pop": summary_func(value)}

    # Set entry and print status
    data_dict[shp.name] = summary_entry(shp, entry, include_latlon)
    if show_status:
        print_status(shp, data_dict, k1, shape_len)

    return data_dict


def raster_clip_weighted(
    raster_weight: Union[str, Path],
    raster_value: Union[str, Path],
    shape_stem: Union[str, Path],
    shape_attr: str = "DOTNAME",
    attr_filter: Union[str, None] = None,
    weight_summary_func: Callable = None,
    include_latlon: bool = False,
) -> dict[str, Union[float, int]]:
    """
    Extracts data from a raster based on shapes.

    Args:
        raster_weight (str): Local path to a raster file used for weights.
        raster_value (str): Local path to a raster file used for values.
        shape_stem (str): Local path stem referencing a set of shape files.
        shape_attr (str): The shape attribute name to be used as the output dictionary key.
        weight_summary_func (Callable): Aggregation function to be used for summarizing clipped data for each shape.
        include_latlon (bool, optional): Flag to include lat/lon in the dictionary entry. Defaults to False.

    Returns:
        dict: A dictionary with dot names as keys and calculated aggregations as values.
    """
    assert Path(raster_weight).is_file(), "Population raster file not found."
    assert Path(raster_value).is_file(), "Values raster file not found."

    # Load data shape and rasters
    shapes = ShapeView.from_file(shape_stem, shape_attr, attr_filter)
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
    """Sum an array and round to the nearest integer."""
    return int(np.round(np.sum(v), 0))


def get_tiff_tags(raster: Image) -> dict[str, Any]:
    """
    Reads tags from a TIFF file.

    Reference:
        https://stackoverflow.com/questions/46477712/reading-tiff-image-metadata-in-python

    Args:
        raster (TIFF): TIFF object.

    Returns:
        dict: A dictionary of tag names and values.
    """
    return {TAGS[t]: raster.tag[t] for t in dict(raster.tag)}


def extract_xy_info_from_raster(raster: Image) -> tuple[float, float, float, float]:
    """
    Extracts x, y, dx, and dy from a raster TIFF file.

    Args:
        raster (TIFF): TIFF object.

    Returns:
    tuple: A tuple of x, y, dx, and dy.
    """

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
    """Initialize a matrix from a raster TIFF file with values > 0"""

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


def subset_matrix_for_clipping(
    shape: ShapeView, sparce_data: np.ndarray, pad: int = 0
) -> np.ndarray:
    """
    Subset the matrix for clipping

    Args:
        shape (ShapeView): Shape object.
        sparce_data (np.ndarray): Sparce matrix of raster data.
        pad (int): Padding for clipping.

    Returns:
        np.ndarray: A subset of the matrix for clipping.
    """
    clip_bool1 = np.logical_and(
        sparce_data[:, 0] > shape.xy_min[0] - pad,
        sparce_data[:, 1] > shape.xy_min[1] - pad,
    )
    clip_bool2 = np.logical_and(
        sparce_data[:, 0] < shape.xy_max[0] + pad,
        sparce_data[:, 1] < shape.xy_max[1] + pad,
    )
    data_clip = sparce_data[np.logical_and(clip_bool1, clip_bool2), :]

    return data_clip


def summary_entry(
    shape: ShapeView, entry: Union[dict, float, int], include_latlon: bool
) -> Union[dict, float, int]:
    """
    Summarize the entry for the shape.

    Args:
        shape (ShapeView): Shape object.
        entry (Union[dict, float, int]): Entry for the shape.
        include_latlon (bool): Flag to include lat/lon in the dictionary entry.

    Returns:
        Union[dict, float, int]: The summarized entry for the shape.
    """
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
    """
    Check if the data is interior to the shape.

    Args:
        shape (ShapeView): Shape object.
        data_clip (np.ndarray): Clipped data.

    Returns:
        bool: True if the data is interior to the shape.
    """
    # Track booleans (indicates if lat/long is interior)
    data_bool = np.zeros(data_clip.shape[0], dtype=bool)

    # Iterate over parts of shapefile
    for path_shp, area_prt in zip(shape.paths, shape.areas):
        # Union of positive areas; intersection with negative areas
        if area_prt > 0:
            data_bool = np.logical_or(
                data_bool, path_shp.contains_points(data_clip[:, :2])
            )
        else:
            data_bool = np.logical_and(
                data_bool, np.logical_not(path_shp.contains_points(data_clip[:, :2]))
            )

    return data_bool


def print_status(shape: ShapeView, data_dict: dict, k1: int, shape_count: int) -> None:
    """Print status message."""
    perc = round(100 * (k1 + 1) / shape_count)
    print(
        k1 + 1,
        "of",
        shape_count,
        f"({perc}%)",
        shape.name,
        shape.center,
        data_dict[shape.name],
    )


def interpolate_at_weight_data(
    shape: ShapeView, weight_clip: np.ndarray, value_clip: np.ndarray, data_bool: bool
) -> float:
    """
    Interpolate at weight data.

    Args:
        shape (ShapeView): Shape object.
        weight_clip (np.ndarray): Clipped weight data.
        value_clip (np.ndarray): Clipped value data.
        data_bool (bool): Boolean indicating if the data is interior to the shape.

    Returns:
        float: The interpolated value at weight data.
    """
    # Calculate population weighted value
    weight = np.sum(weight_clip[data_bool, 2])

    # Prep interpolate coordinates and value arguments
    value_args = [value_clip[:, 0:2], value_clip[:, 2]]

    if weight > 0:
        # Interpolate at weight, assign -1 for problems
        val_est = interpolate.griddata(*value_args, weight_clip[:, 0:2], fill_value=-1)
        if -1 in val_est:
            err_dex = val_est == -1
            # Use the nearest value for problems
            val_rev = interpolate.griddata(
                *value_args, weight_clip[err_dex, 0:2], method="nearest"
            )
            val_est[err_dex] = val_rev
        # Use population to weight values
        final_val = np.sum(weight_clip[data_bool, 2] * val_est[data_bool]) / weight
    else:
        # No population data, interpolate at boundary, assign -1 for problems
        val_est = interpolate.griddata(*value_args, shape.points[:, 0:2], fill_value=-1)
        if -1 in val_est:
            err_dex = val_est == -1
            # Use the nearest value for problems
            val_rev = interpolate.griddata(
                *value_args, shape.points[err_dex, 0:2], method="nearest"
            )
            val_est[err_dex] = val_rev

        # Average values at shape perimeter
        final_val = np.mean(val_est)

    return final_val
