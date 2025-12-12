"""
Functions for spatial processing of shape files.
"""

from __future__ import annotations

import itertools
import matplotlib.path as plth
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
import tempfile

from pathlib import Path
from pyproj import Geod
from shapefile import Shape, ShapeRecord, Reader, Shapes, Writer, POINT
from shapely.geometry import Polygon, MultiPolygon, LinearRing, Point
from shapely.prepared import prep
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi

from typing import Union


class ShapeView:
    """Class extracting and encapsulating shape data used for raster processing."""

    default_shape_attr: str = "DOTNAME"

    def __init__(self, shape: Shape, record: ShapeRecord, name_attr: str = None):
        self.name_attr: str = name_attr or self.default_shape_attr
        self.shape: Shape = shape
        self._points: np.ndarray = None
        self.record: ShapeRecord = record
        self.center: tuple[float, float] = (0.0, 0.0)
        self.paths: list[plth.Path] = []
        self.areas: list[float] = []

    def __str__(self):
        """String representation used to print or debug WeatherSet objects."""
        return f"{self.name} (parts: {str(len(self.areas))})"

    @property
    def name(self):
        """Shape name, read using name attribute."""
        return self.record[self.name_attr]

    @property
    def points(self):
        """The list of point defining shape geometry."""
        if self._points is None:
            self._points = np.array(self.shape.points)
        return self._points

    @property
    def xy_max(self):
        """Max x, y coordinates, based on point coordinates."""
        return np.max(self.points, axis=0)

    @property
    def xy_min(self):
        """Min x, y coordinates, based on point coordinates."""
        return np.min(self.points, axis=0)

    @property
    def parts_count(self):
        """Number of shape parts."""
        return len(self.paths)

    def validate(self) -> None:
        assert self.points.shape[0] != 0 and len(self.paths) > 0, "No parts in a shape."
        assert len(self.paths) == len(self.areas), "Inconsistent number of parts in a shape."
        assert self.name is not None and self.name != "", "Shape has no name."

    def as_polygon(self) -> Polygon:
        return shapely.geometry.shape(self.shape)

    def as_multi_polygon(self) -> MultiPolygon:
        return self._as_multi_polygon(self.as_polygon())

    @property
    def area_km2(self):
        return polygon_area_km2(self.as_polygon())

    @staticmethod
    def _as_multi_polygon(shape: Shape):
        return MultiPolygon([shape]) if isinstance(shape, Polygon) else shape

    @classmethod
    def read_shapes(cls,
                    shape_stem: Union[str, Path, Reader]
                    ) -> tuple[Reader, Shapes[Shape], list[ShapeRecord]]:
        reader: Reader = shape_stem if isinstance(shape_stem, Reader) else Reader(str(shape_stem))
        shapes: Shapes[Shape] = reader.shapes()
        records: list[ShapeRecord] = reader.records()
        print(reader.fields)
        return reader, shapes, records

    @classmethod
    def from_file(cls,
                  shape_stem: Union[str, Path, Reader],
                  shape_attr: Union[str, None] = None,
                  attr_filter: Union[str, None] = None,
                  ) -> list[ShapeView]:
        """
        Loads a shape into a shape view class.

        Args:
            shape_stem (str): Local path stem referencing a set of shape files.
            shape_attr (str): The shape attribute name to be used as the output dictionary key.

        Returns:
            list: A list of `ShapeView` objects containing parsed shape information.
        """
        # Shapefiles
        reader, sf1s, sf1r = cls.read_shapes(shape_stem)

        # Output dictionary
        shapes_data: list[cls] = []

        # Iterate of shapes in shapefile
        for k1 in range(len(sf1r)):
            # First (only) field in shapefile record is dot-name
            shp = cls(shape=sf1s[k1], record=sf1r[k1], name_attr=shape_attr)

            # Only retain shapes with specified name
            if (attr_filter and not shp.name.startswith(attr_filter)):
                continue

            # List of parts in (potentially) multi-part shape
            prt_list = list(shp.shape.parts) + [len(shp.points)]

            # Accumulate total area centroid over multiple parts
            Cx_tot = Cy_tot = Axy_tot = 0.0

            # Iterate over parts of shapefile
            for k2 in range(len(prt_list) - 1):
                shp_prt = shp.points[prt_list[k2]:prt_list[k2 + 1]]
                path_shp = plth.Path(shp_prt, closed=True, readonly=True)

                # Estimate area for part
                area_prt = area_sphere(shp_prt)

                shp.paths.append(path_shp)
                shp.areas.append(area_prt)

                # Estimate area centroid for part, accumulate
                (Cx, Cy, Axy) = centroid_area(shp_prt)
                Cx_tot += Cx * Axy
                Cy_tot += Cy * Axy
                Axy_tot += Axy

            # Update value for area centroid
            shp.center = (Cx_tot / Axy_tot, Cy_tot / Axy_tot)

            shapes_data.append(shp)

        return shapes_data


# Helpers


def shapes_to_polygons_dict(
    shape_stem: Union[str, Path, Reader], all_multi: bool = True
) -> list[MultiPolygon]:
    """
    Converts shapes from a shapefile into a dictionary of MultiPolygons.

    Args:
        shape_stem (Union[str, Path, Reader]): The path or identifier for the shapefile.
        all_multi (bool, optional): If True, ensures all geometries are MultiPolygons. Defaults to True.

    Returns:
        list[MultiPolygon]: A dictionary where keys are shape identifiers and values are MultiPolygon objects.
    """
    # Example loading shape files as multi polygons
    # https://gis.stackexchange.com/questions/70591/creating-shapely-multipolygons-from-shapefile-multipolygons
    _, shapes, records = ShapeView.read_shapes(shape_stem)
    polygons = {r.DOTNAME: shapely.geometry.shape(s) for s, r in zip(shapes, records)}
    if all_multi:
        polygons = {
            n: MultiPolygon([p]) if isinstance(p, Polygon) else p
            for n, p in polygons.items()
        }

    return polygons


def shapes_to_polygons(
    shape_stem: Union[str, Path, Reader], all_multi: bool = True
) -> list[MultiPolygon]:
    """
    Converts shapes from a shapefile into a list of MultiPolygons.

    Args:
        shape_stem (Union[str, Path, Reader]): The path or identifier for the shapefile.
        all_multi (bool, optional): If True, ensures all geometries are MultiPolygons. Defaults to True.

    Returns:
        list[MultiPolygon]: A list of MultiPolygon objects.
    """
    d = shapes_to_polygons_dict(shape_stem=shape_stem, all_multi=all_multi)
    return list(d.values())


def polygon_contains(
    polygon: Union[Polygon, MultiPolygon], points: Union[np.ndarray, list[Point]]
) -> np.ndarray:
    """
    Determines which points are inside a polygon.

    Args:
        polygon (Union[Polygon, MultiPolygon]): The polygon to check.
        points (Union[np.ndarray, list[Point]]): The points to check.

    Returns:
        np.ndarray: An array of points that are inside the polygon.
    """
    mp = prep(polygon)  # prep
    pts: list[Point] = (
        [Point(t[0], t[1]) for t in points]
        if isinstance(points, np.ndarray)
        else points
    )
    pts_in = [p for p in pts if mp.contains(p)]
    pts_in_array: np.ndarray = np.array([[p.x, p.y] for p in pts_in])
    return pts_in_array


def polygon_area_km2(polygon: Union[Polygon, MultiPolygon]) -> np.float64:
    """
    Calculates the area of a polygon in square kilometers.

    Args:
        polygon (Union[Polygon, MultiPolygon]): The polygon to calculate the area for.

    Returns:
        np.float64: The area of the polygon in square kilometers
    """
    geod = Geod(ellps="WGS84")
    area, _ = geod.geometry_area_perimeter(polygon)  # perimeter ignored
    area_km2 = np.float64(abs(area)) / 1000000.0
    return area_km2


def polygon_to_coords(geom: Union[Polygon, LinearRing]) -> list[tuple[float, float]]:
    """
    Converts a polygon or linear ring to a list of coordinates.

    Args:
        geom (Union[Polygon, LinearRing]): The polygon or linear ring to convert.

    Returns:
        list[tuple[float, float]]: A list of coordinates.

    """
    if isinstance(geom, Polygon):
        xy_set = geom.exterior.coords
    elif isinstance(geom, LinearRing):
        xy_set = geom.coords
    else:
        raise TypeError(f"Unsupported geometry type {type(geom)}")

    shp_prt: np.ndarray = np.array([(val[0], val[1]) for val in xy_set])
    coords_list: list[tuple[float, float]] = shp_prt.tolist()
    return coords_list


def polygons_to_parts(polygons: list[Polygon]) -> list[list[tuple[float, float]]]:
    """
    Converts a list of polygons to a list of parts.

    Args:
        polygons (list[Polygon]): A list of polygons to convert.

    Returns:
        list[list[tuple[float, float]]]: A list of parts
    """

    all_polygons = [[p] + list(p.interiors) for p in polygons]
    all_polygons_list = list(itertools.chain(*all_polygons))
    poly_as_list = [polygon_to_coords(p) for p in all_polygons_list]
    return poly_as_list


def area_sphere(shape_points) -> float:
    """
    Calculates the area of a polygon on a sphere.

    Reference:
        JGeod (2013) v87 p43-55

    Args:
        shape_points (numpy.ndarray): A (N,2) numpy array representing a shape
            (first point equals last point, clockwise direction is positive).

    Returns:
        float: The area of the polygon.
    """
    sp_rad = np.radians(shape_points)
    beta1 = sp_rad[:-1, 1]
    beta2 = sp_rad[1:, 1]
    domeg = sp_rad[1:, 0] - sp_rad[:-1, 0]

    val1 = np.tan(domeg / 2) * np.sin((beta2 + beta1) / 2.0) * np.cos((beta2 - beta1) / 2.0)
    dalph = 2.0 * np.arctan(val1)
    tarea = 6371.0 * 6371.0 * np.sum(dalph)

    return tarea


def centroid_area(shape_points) -> tuple[float, float, float]:
    """
    Calculates the area centroid of a polygon based on Cartesian coordinates.

    Note:
        The area calculated by this function is not a good estimate for a spherical polygon
        and should only be used in weighting multi-part shape centroids.

    Args:
        shape_points (numpy.ndarray): A (N,2) numpy array representing a shape
            (first point equals last point, clockwise direction is positive).

    Returns:
        tuple: A tuple containing the centroid coordinates and area as floats (Cx, Cy, A).
    """

    a_vec = shape_points[:-1, 0] * shape_points[1:, 1] - shape_points[1:, 0] * shape_points[:-1, 1]

    A = np.sum(a_vec) / 2.0
    Cx = np.sum((shape_points[:-1, 0] + shape_points[1:, 0]) * a_vec) / 6.0 / A
    Cy = np.sum((shape_points[:-1, 1] + shape_points[1:, 1]) * a_vec) / 6.0 / A

    return (Cx, Cy, A)


def long_mult(lat):  # latitude in degrees
    """Returns the multiplier for longitude based on latitude."""
    return 1.0 / np.cos(lat * np.pi / 180.0)


# API


def shape_subdivide(
    shape_stem: Union[str, Path],
    out_dir: Union[str, Path] = None,
    out_suffix: str = None,
    output_centers: bool = False,
    top_n: int = None,
    shape_attr: str = "DOTNAME",
    box_target_area_km2: int = None,
    points_per_box: int = None,
    random_seed: int = None,
    verbose: bool = False,
) -> str:
    """
    Creates a new shapefile that subdivides the original shapes based on area (unweighted) or population (weighted).

    Args:
        shape_stem (str): Local shape file path or stem (path without extension).
        out_dir (str, optional): Local directory where outputs are stored. Defaults to a new temporary directory.
        out_suffix (str, optional): Suffix of the output stem. Defaults to a suffix containing `box_target_area_km2`.
        output_centers (bool, optional): Flag controlling whether to export sub-shape centers. Defaults to False.
        top_n (int, optional): Number of top MultiPolygons to process. Used for testing large datasets.
            By default, all MultiPolygons are processed.
        shape_attr (str, optional): The shape's attribute used as a prefix for the output shape's identity attribute.
            Defaults to "DOTNAME".
        box_target_area_km2 (float, optional): Target box area used to calculate the number of boxes (clusters).
        points_per_box (int, optional): Points-per-box dimension. Higher values result in slower but more accurate processing.
        random_seed (int, optional): Random seed for reproducibility.
        verbose (bool, optional): Whether to show debug information.

    Returns:
        str: Local path prefix (output shapes stem).
    """

    shape_stem = Path(shape_stem)
    if shape_stem.suffix != "":
        shape_stem = shape_stem.with_suffix("")

    box_target_area_km2 = box_target_area_km2 or 100
    points_per_box = points_per_box or 250
    random_seed = random_seed or 4

    assert box_target_area_km2 > 0, (
        "Argument 'box_target_area_km2' must be a positive integer."
    )
    assert points_per_box > 0, "Argument 'points_per_box' must be a positive integer."
    assert random_seed > 0, "Argument 'random_seed' must be a positive integer."

    # Read shapes
    sf1 = Reader(shape_stem)
    multi_list = shapes_to_polygons(sf1)
    rec_list = sf1.records()

    # Create shape writer
    out_dir = Path(out_dir or Path(tempfile.mkdtemp()))
    out_suffix = out_suffix or f"sub_{box_target_area_km2}km"
    out_shape_name = f"{Path(shape_stem).name}_{out_suffix}"
    out_shape_stem = Path(out_dir.joinpath(out_shape_name))

    out_shape_stem.parent.mkdir(exist_ok=True, parents=True)
    sf1new = Writer(out_shape_stem)
    sf1new.field(shape_attr, "C", 70, 0)
    sf1new.fields.extend(
        [tuple(t) for t in sf1.fields if t[0] not in ["DeletionFlag", shape_attr]]
    )

    if output_centers:
        sf1new2 = Writer(f"{out_shape_stem}_centers", shapeType=POINT)
        sf1new2.field(shape_attr, "C", 70, 0)
    else:
        sf1new2 = None

    assert (shape_attr in [f[0] for f in sf1new.fields]), f"Shape doesn't contain {shape_attr} field."

    # Second step is to create an underlying mesh of points. If the mesh is
    # equidistant, then the subdivided shapes will be uniform area. Alternatively,
    # the points could be population raster data, and the subdivided shapes would
    # be uniform population.

    top_n = top_n or len(multi_list)

    for k1, multi in enumerate(multi_list[:top_n]):
        multi_area = polygon_area_km2(multi)
        num_box = np.maximum(int(np.round(multi_area / box_target_area_km2)), 1)
        pts_dim = int(np.ceil(np.sqrt(points_per_box * num_box)))

        if not multi.is_valid:
            multi = multi.buffer(0)  # this seems to be fixing broken multi-polygons.
            if verbose and multi.is_valid:
                print(f"Fixed the invalid MultiPolygon {k1}.")

        if multi.is_valid:
            # Debug logging: shapefile index, target number of subdivisions
            bounds_str = str([round(v, 2) for v in multi.bounds])
            if verbose:
                print(
                    f"MultiPolygon: {k1:<5} {bounds_str:<32} Number of boxes: {num_box}"
                )
        else:
            Warning(f"Unable to fix the MultiPolygon {k1}!")

        # Start with a rectangular mesh, then (roughly) correct longitude (x values);
        # Assume spacing on latitude (y values) is constant; x value spacing needs to
        # be increased based on y value.
        xspan = [multi.bounds[0], multi.bounds[2]]
        yspan = [multi.bounds[1], multi.bounds[3]]
        xcv, ycv = np.meshgrid(
            np.linspace(xspan[0], xspan[1], pts_dim),
            np.linspace(yspan[0], yspan[1], pts_dim),
        )

        pts_vec = np.zeros((pts_dim * pts_dim, 2))
        pts_vec[:, 0] = np.reshape(xcv, pts_dim * pts_dim)
        pts_vec[:, 1] = np.reshape(ycv, pts_dim * pts_dim)
        pts_vec[:, 0] = pts_vec[:, 0] * long_mult(pts_vec[:, 1]) - xspan[0] * (
            long_mult(pts_vec[:, 1]) - 1
        )

        # Same idea here as in raster clipping; identify points that are inside the shape
        # and keep track of them using inBool
        pts_vec_in = polygon_contains(multi, pts_vec)

        # Feed points interior to shape into k-means clustering to get num_box equal(-ish) clusters;
        sub_clust = KMeans(
            n_clusters=num_box, random_state=random_seed, n_init="auto"
        ).fit(pts_vec_in)
        sub_node = (
            sub_clust.cluster_centers_
        )  # this is not a bug, that is the actual name of the property

        # Don't actually want the cluster centers, goal is the outlines. Going from centers
        # to outlines uses Voronoi tessellation. Add a box of external points to avoid mucking
        # up the edges. (+/- 200 was arbitrary value greater than any possible lat/long)
        assert max(abs(sub_node.reshape(-1))) < 200, "Coordinates must be < 200."
        EXT_PTS = np.array([[-200, -200], [200, -200], [-200, 200], [200, 200]])
        vor_node = np.append(sub_node, EXT_PTS, axis=0)
        vor_obj = Voronoi(vor_node)

        # Extract the Voronoi region boundaries from the Voronoi object. Need to duplicate
        # first point in each region so last == first for the next step
        vor_list = list()
        vor_vert = vor_obj.vertices
        for vor_reg in vor_obj.regions:
            if -1 in vor_reg or len(vor_reg) == 0:
                continue
            vor_loop = np.append(
                vor_vert[vor_reg, :], vor_vert[vor_reg[0:1], :], axis=0
            )
            vor_list.append(vor_loop)

        # If there's not 1 Voronoi region outline for each k-means cluster center
        # at this point, something has gone very wrong. Time to bail.
        if len(vor_list) != len(sub_node):
            raise ValueError(
                "Failed to create a Voronoi region outline for each k-means cluster center."
            )

        # The Voronoi region outlines may extend beyond the shape outline and/or
        # overlap with negative spaces, so intersect each Voronoi region with the
        # shapely MultiPolygon created previously
        new_recs = None
        for k2, poly in enumerate(vor_list):
            # Voronoi region are convex, so will not need MultiPolygon object
            poly_reg = (Polygon(poly)).intersection(multi)

            # Each Voronoi region will be a new shape; give it a name
            new_recs = rec_list[k1].as_dict()
            dotname = rec_list[k1][shape_attr]
            dotname_new = f"{dotname}:A{k2:04d}"
            new_recs[shape_attr] = dotname_new

            assert poly_reg.geom_type in ["Polygon", "MultiPolygon"], (
                "Unsupported geometry type"
            )
            poly_list = (
                poly_reg.geoms if poly_reg.geom_type == "MultiPolygon" else [poly_reg]
            )
            poly_as_list = polygons_to_parts(poly_list)

            # Add the new shape to the shapefile; splat the record
            sf1new.poly(poly_as_list)
            sf1new.record(**new_recs)

        if output_centers and new_recs is not None:
            for i, p in enumerate([Point(xy) for xy in sub_node]):
                sf1new2.point(p.x, p.y)
                assert output_centers
                sf1new2.record(*new_recs)

    sf1new.close()
    if output_centers:
        sf1new2.close()

    return str(out_shape_stem)


def plot_shapes(
    shape_stem: Union[str, Path],
    ax: plt.Axes = None,
    alpha: float = 1.0,
    color: Union[str, None] = None,
    linewidth: float = 1,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots shapes from a shapefile.

    Args:

        shape_stem (Union[str, Path]): The path or identifier for the shapefile.
        ax (plt.Axes, optional): The axis to plot the shapes on. Defaults to None.
        alpha (float, optional): The transparency of the shapes. Defaults to 1.0.
        color (Union[str, None], optional): The color of the shapes. Defaults to None.
        linewidth (float, optional): The width of the line. Defaults to 1.
        **kwargs: Additional keyword arguments for the plot.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axis objects.
    """
    # Plot sub-shapes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if color is not None:
        kwargs["facecolor"] = color
    kwargs["alpha"] = alpha
    kwargs["linewidth"] = linewidth

    multi_list: list[MultiPolygon] = shapes_to_polygons(shape_stem)
    x_min, x_max, y_min, y_max = 360.0, -360.0, 90.0, -90.0
    for multi in multi_list:
        for poly in multi.geoms:
            x, y = poly.exterior.xy
            x_min, x_max = min(x_min, min(x)), max(x_max, max(x))
            y_min, y_max = min(y_min, min(y)), max(y_max, max(y))
            ax.fill(x, y, **kwargs)

    # Set the axis limits and show the plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return fig, ax


# Plot generated shapes into a file
def plot_subdivision(
    shape_file: Union[str, Path],
    subdivision_stam: Union[str, Path],
    shape_color: str = "gray",
    subdivision_color: str = "red",
    png_dpi=1800,
) -> None:
    """
    Plots shapes and their subdivisions into a PNG file.

    Args:
        shape_file (Union[str, Path]): The path or identifier for the shapefile.
        subdivision_stam (Union[str, Path]): The path or identifier for the subdivision shapefile.
        shape_color (str, optional): The color of the shapes. Defaults to "gray".
        subdivision_color (str, optional): The color of the subdivisions. Defaults to "red".
        png_dpi (int, optional): The DPI of the PNG file. Defaults to 1800.

    Returns:
        None
    """
    png_file = Path(subdivision_stam).with_suffix(".png")
    fig, ax = plot_shapes(
        shape_file, alpha=0.5, color=None, linewidth=1.0, edgecolor=shape_color
    )
    plot_shapes(
        subdivision_stam,
        ax=ax,
        color="None",
        alpha=0.3,
        linewidth=0.2,
        edgecolor=subdivision_color,
    )
    fig.savefig(png_file, dpi=png_dpi)

    return None
