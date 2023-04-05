"""
Functions for spatial processing of shape files.
"""

from __future__ import annotations

import itertools
import matplotlib.path as plth
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry

from pathlib import Path
from pyproj import Geod
from shapefile import Shape, ShapeRecord, Reader, Shapes, Writer, POINT
from shapely.geometry import Polygon, MultiPolygon, LinearRing, Point
from shapely.prepared import prep
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi

from typing import Any, Dict, List, Tuple, Union, Callable


class ShapeView:
    """Class extracting and encapsulating shape data used for raster processing."""
    default_shape_attr: str = "DOTNAME"

    def __init__(self, shape: Shape, record: ShapeRecord, name_attr: str = None):
        self.name_attr: str = name_attr or self.default_shape_attr
        self.shape: Shape = shape
        self.record: ShapeRecord = record
        self.center: (float, float) = (0.0, 0.0)
        self.paths: List[plth.Path] = []
        self.areas: List[float] = []

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
        return np.array(self.shape.points)

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
    def read_shapes(cls, shape_stem: Union[str, Path, Reader]) -> Tuple[Reader, Shapes[Shape], List[ShapeRecord]]:
        reader: Reader = shape_stem if isinstance(shape_stem, Reader) else Reader(str(shape_stem))
        shapes: Shapes[Shape] = reader.shapes()
        records: List[ShapeRecord] = reader.records()
        return reader, shapes, records

    @classmethod
    def from_file(cls, shape_stem: Union[str, Path, Reader], shape_attr: Union[str, None] = None) -> List[ShapeView]:
        """
        Load shape into a shape view class.
        :param shape_stem: Local path stem referencing a set of shape files.
        :param shape_attr: The shape attribute name to be use as output dictionary key.
        :return: List of ShapeView objects, containing parsed shape info.
        """
        # Shapefiles
        reader, sf1s, sf1r = cls.read_shapes(shape_stem)

        # Output dictionary
        shapes_data: List[cls] = []

        # Iterate of shapes in shapefile
        for k1 in range(len(sf1r)):
            # First (only) field in shapefile record is dot-name
            shp = cls(shape=sf1s[k1], record=sf1r[k1], name_attr=shape_attr)

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

def shapes_to_polygons_dict(shape_stem: Union[str, Path, Reader], all_multi: bool = True) -> List[MultiPolygon]:
    # Example loading shape files as multi polygons
    # https://gis.stackexchange.com/questions/70591/creating-shapely-multipolygons-from-shapefile-multipolygons
    _, shapes, records = ShapeView.read_shapes(shape_stem)
    polygons = {r.DOTNAME: shapely.geometry.shape(s) for s, r in zip(shapes, records)}
    if all_multi:
        polygons = {n: MultiPolygon([p]) if isinstance(p, Polygon) else p for n, p in polygons.items()}

    return polygons


def shapes_to_polygons(shape_stem: Union[str, Path, Reader], all_multi: bool = True) -> List[MultiPolygon]:
    d = shapes_to_polygons_dict(shape_stem=shape_stem, all_multi=all_multi)
    return list(d.values())


def polygon_contains(polygon: Union[Polygon, MultiPolygon],
                            points: Union[np.ndarray, List[Point]]) -> np.ndarray:
    mp = prep(polygon)  # prep
    pts: List[Point] = [Point(t[0], t[1]) for t in points] if isinstance(points, np.ndarray) else points
    pts_in = [p for p in pts if mp.contains(p)]
    pts_in_array: np.ndarray = np.array([[p.x, p.y] for p in pts_in])
    return pts_in_array


def polygon_area_km2(polygon: Union[Polygon, MultiPolygon]) -> np.float64:
    geod = Geod(ellps="WGS84")
    area, _ = geod.geometry_area_perimeter(polygon)  # perimeter ignored
    area_km2 = np.float64(abs(area))/1000000.0
    return area_km2


def polygon_to_coords(geom: Union[Polygon, LinearRing]) -> List[Tuple[float, float]]:
    if isinstance(geom, Polygon):
        xy_set = geom.exterior.coords
    elif isinstance(geom, LinearRing):
        xy_set = geom.coords
    else:
        raise TypeError(f"Unsupported geometry type {type(geom)}")

    shp_prt: np.ndarray = np.array([(val[0], val[1]) for val in xy_set])
    coords_list: List[Tuple[float, float]] = shp_prt.tolist()
    return coords_list


def polygons_to_parts(polygons: List[Polygon]) -> List[List[Tuple[float, float]]]:
    all_polygons = [[p] + list(p.interiors) for p in polygons]
    all_polygons_list = list(itertools.chain(*all_polygons))
    poly_as_list = [polygon_to_coords(p) for p in all_polygons_list]
    return poly_as_list


def area_sphere(shape_points) -> float:
    """
    Calculates Area of a polygon on a sphere; JGeod (2013) v87 p43-55
    :param shape_points: point (N,2) numpy array representing a shape (first == last point, clockwise == positive)
    :return: shape area as a float
    """
    sp_rad = np.radians(shape_points)
    beta1 = sp_rad[:-1, 1]
    beta2 = sp_rad[1:, 1]
    domeg = sp_rad[1:, 0] - sp_rad[:-1, 0]

    val1 = np.tan(domeg / 2) * np.sin((beta2 + beta1) / 2.0) * np.cos((beta2 - beta1) / 2.0)
    dalph = 2.0 * np.arctan(val1)
    tarea = 6371.0 * 6371.0 * np.sum(dalph)

    return tarea


def centroid_area(shape_points) -> (float, float, float):
    """
    Calculates the area centroid of a polygon based on cartesean coordinates.
    Area calculated by this function is not a good estimate for a spherical
    polygon, and should only be used in weighting multi-part shape centroids.
    :param shape_points: point (N,2) numpy array representing a shape
                        (first == last point, clockwise == positive)
    :return: (Cx, Cy, A) Coordinates and area as floats
    """

    a_vec = (shape_points[:-1, 0] * shape_points[1:, 1] -
             shape_points[1:, 0] * shape_points[:-1, 1])

    A = np.sum(a_vec) / 2.0
    Cx = np.sum((shape_points[:-1, 0] + shape_points[1:, 0]) * a_vec) / 6.0 / A
    Cy = np.sum((shape_points[:-1, 1] + shape_points[1:, 1]) * a_vec) / 6.0 / A

    return (Cx, Cy, A)


def long_mult(lat): # latitude in degrees
  return 1.0/np.cos(lat*np.pi/180.0)


# API


def shape_subdivide(shape_stem: Union[str, Path],
                    out_shape_stem: Union[str, Path] = None,
                    out_centers: bool = False,
                    top_n: int = None,
                    shape_attr: str = "DOTNAME") -> None:

    # Read shapes, convert to multi polygonsvor_list
    sf1 = Reader(shape_stem)
    multi_list = shapes_to_polygons(sf1)
    rec_list = sf1.records()

    # Create shape writer
    out_shape_stem = out_shape_stem or f"{str(shape_stem)}_sub"
    out_shape_stem = Path(out_shape_stem)
    out_shape_stem.parent.mkdir(exist_ok=True, parents=True)
    sf1new = Writer(out_shape_stem)
    sf1new.field(shape_attr, 'C', 70, 0)
    sf1new.fields.extend([tuple(t) for t in sf1.fields if t[0] not in ["DeletionFlag", shape_attr]])

    if out_centers:
        sf1new2 = Writer(f"{out_shape_stem}_centers", shapeType=POINT)
        sf1new2.field(shape_attr, 'C', 70, 0)
        #sf1new2.field("ID", "N", 10)
    else:
        sf1new2 = None

    field_names = [f[0] for f in sf1new.fields]
    assert shape_attr in field_names, f"Shape doesn't contain {shape_attr} field."
    dotname_index = field_names.index(shape_attr)


    # Second step is to create an underlying mesh of points. If the mesh is
    # equidistant, then the subdivided shapes will be uniform area. Alternatively,
    # the points could be population raster data, and the subdivided shapes would
    # be uniform population.

    top_n = top_n or len(multi_list)
    # multi_count = len(multi_list)
    # if top_n is not None:
    #     assert isinstance(top_n, int) and top_n > 0, "Argument top_n must be a positive integer."
    #     multi_count = abs(min(multi_count, top_n))

    for k1, multi in enumerate(multi_list[:top_n]):
        AREA_TARG = 100  # Needs to be configurable; here target is ~100km^2
        PPB_DIM = 250  # Points-per-box-dimension; tuning; higher is slower and more accurate
        RND_SEED = 4    # Random seed; ought to expose for reproducibility

        multi_area = polygon_area_km2(multi)
        num_box = np.maximum(int(np.round(multi_area/AREA_TARG)), 1)
        pts_dim = int(np.ceil(np.sqrt(PPB_DIM*num_box)))

        if not multi.is_valid:
            print(k1, f"Trying to fix the invalid Multipolygon {k1}.")
            multi = multi.buffer(0)  # this seems to be fixing broken multi-polygons.

        # If the multi polygoin isn't valid; need to bail
        if not multi.is_valid:
            print(k1, f"Multipolygon {k1} not valid")
            continue
        else:
            # Debug logging: shapefile index, target number of subdivisions
            print(k1, num_box)

        # Start with a rectangular mesh, then (roughly) correct longitude (x values);
        # Assume spacing on latitude (y values) is constant; x value spacing needs to
        # be increased based on y value.
        xspan = [multi.bounds[0], multi.bounds[2]]
        yspan = [multi.bounds[1], multi.bounds[3]]
        xcv, ycv = np.meshgrid(np.linspace(xspan[0], xspan[1], pts_dim),
                               np.linspace(yspan[0], yspan[1], pts_dim))

        pts_vec = np.zeros((pts_dim*pts_dim, 2))
        pts_vec[:, 0] = np.reshape(xcv, pts_dim*pts_dim)
        pts_vec[:, 1] = np.reshape(ycv, pts_dim*pts_dim)
        pts_vec[:, 0] = pts_vec[:, 0] * long_mult(pts_vec[:, 1]) - xspan[0]*(long_mult(pts_vec[:, 1]) - 1)

        # Same idea here as in raster clipping; identify points that are inside the shape
        # and keep track of them using inBool
        pts_vec_in = polygon_contains(multi, pts_vec)

        # Feed points interior to shape into k-means clustering to get num_box equal(-ish) clusters;
        sub_clust = KMeans(n_clusters=num_box, random_state=RND_SEED, n_init='auto').fit(pts_vec_in)
        sub_node = sub_clust.cluster_centers_  # this is not a bug, that is the actual name of the property

        # Don't actually want the cluster centers, goal is the outlines. Going from centers
        # to outlines uses Voronoi tessellation. Add a box of external points to avoid mucking
        # up the edges. (+/- 200 was arbitrary value greater than any possible lat/long)
        EXT_PTS = np.array([[-200, -200], [ 200, -200], [-200, 200], [200, 200]])
        vor_node = np.append(sub_node, EXT_PTS, axis=0)
        vor_obj = Voronoi(vor_node)

        # Extract the Voronoi region boundaries from the Voronoi object. Need to duplicate
        # first point in each region so last == first for the next step
        vor_list = list()
        vor_vert = vor_obj.vertices
        for vor_reg in vor_obj.regions:
            if -1 in vor_reg or len(vor_reg) == 0:
                continue
            vor_loop = np.append(vor_vert[vor_reg, :], vor_vert[vor_reg[0:1], :], axis=0)
            vor_list.append(vor_loop)

        # If there's not 1 Voronoi region outline for each k-means cluster center
        # at this point, something has gone very wrong. Time to bail.
        if len(vor_list) != len(sub_node):
            print(k1, 'BLARG')
            continue

        # The Voronoi region outlines may extend beyond the shape outline and/or
        # overlap with negative spaces, so intersect each Voronoi region with the
        # shapely MultiPolygon created previously
        for k2, poly in enumerate(vor_list):
            # Voronoi region are convex, so will not need MultiPolygon object
            poly_reg = (Polygon(poly)).intersection(multi)

            # Each Voronoi region will be a new shape; give it a name
            new_recs = list(rec_list[k1]).copy()
            dotname = rec_list[k1][dotname_index]
            dotname_new = f"{dotname}:A{k2:04d}"
            new_recs[dotname_index] = dotname_new

            assert poly_reg.geom_type in ["Polygon", "MultiPolygon"], "Unsupported geometry type"
            poly_list = poly_reg.geoms if poly_reg.geom_type == "MultiPolygon" else [poly_reg]
            poly_as_list = polygons_to_parts(poly_list)

            # Add the new shape to the shapefile; splat the record
            sf1new.poly(poly_as_list)
            sf1new.record(*new_recs)

        if out_centers:
            for i, p in enumerate([Point(xy) for xy in sub_node]):
                sf1new2.point(p.x, p.y)
                assert out_centers
                sf1new2.record(*new_recs)

    sf1new.close()
    if out_centers:
        sf1new2.close()


def plot_shapes(shape_stem: Union[str, Path],
                ax: plt.Axes = None,
                alpha: float = 0.5,
                color: str = "red",
                line_width: float = 1) -> Tuple[plt.Figure, plt.Axes]:

    # Plot sub-shapes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    multi_list: List[MultiPolygon] = shapes_to_polygons(shape_stem)
    x_min, x_max, y_min, y_max = -360.0, 360.0, -90.0, 90.0
    for multi in multi_list:
        for poly in multi.geoms:
            x, y = poly.exterior.xy
            x_min, x_max = max(x_min, min(x)), min(x_max, max(x))
            y_min, y_max = max(y_min, min(y)), min(y_max, max(y))
            ax.fill(x, y, alpha=alpha, linewidth=line_width)
            ax.fill(x, y, alpha=alpha, linewidth=line_width, color=color)

    # Set the axis limits and show the plot
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    return fig, ax
