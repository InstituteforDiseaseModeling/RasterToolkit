"""
Functions for spatial processing of shape files.
"""

from __future__ import annotations

import matplotlib.path as plt
import numpy as np

from pathlib import Path

import shapely.geometry
from shapefile import Shape, ShapeRecord, Reader, Shapes, Writer
from shapely.geometry import Polygon, MultiPolygon, Point
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
        self.paths: List[plt.Path] = []
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

    @staticmethod
    def _as_multi_polygon(shape: Shape):
        return MultiPolygon([shape]) if isinstance(shape, Polygon) else shape

    @classmethod
    def _read(cls, shape_stem: Union[str, Path, Reader]) -> Tuple[Reader, Shapes[Shape], List[ShapeRecord]]:
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
        reader, sf1s, sf1r = cls._read(shape_stem)

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
                path_shp = plt.Path(shp_prt, closed=True, readonly=True)

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

    @classmethod
    def to_multi_polygons(cls, shape_stem: Union[str, Path, Reader]) -> List[MultiPolygon]:
        """
        Load shape into a shape view class.
        :param shape_stem: Local path stem referencing a set of shape files.
        :return: List of shapely MultiPolygon objects
        """
        # Example loading shape files as multi polygons
        # https://gis.stackexchange.com/questions/70591/creating-shapely-multipolygons-from-shapefile-multipolygons
        _, shapes, _ = cls._read(shape_stem)
        polis = [shapely.geometry.shape(s) for s in shapes]
        multis = [MultiPolygon([p]) if isinstance(p, Polygon) else p for p in polis]
        return multis


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


def shape_subdivide(shape_stem: Union[str, Path], out_shape_stem: Union[str, Path], shape_attr: str = "DOTNAME"):
    # Read shapes, convert to multi polygonsvor_list
    sf1 = Reader(shape_stem)
    multi_list = ShapeView.to_multi_polygons(sf1)

    # Create shape writer
    Path(out_shape_stem).mkdir(exist_ok=True, parents=True)
    sf1new = Writer(out_shape_stem)
    sf1new.field('DOTNAME', 'C', 70, 0)
    sf1new.fields.extend([tuple(t) for t in sf1.fields if t[0] not in ["DeletionFlag", "DOTNAME"]])

    # Second step is to create an underlying mesh of points. If the mesh is
    # equidistant, then the subdivided shapes will be uniform area. Alternatively,
    # the points could be population raster data, and the subdivided shapes would
    # be uniform population.

    for k1, multi in enumerate(multi_list):
        AREA_TARG = 100  # Needs to be configurable; here target is ~100km^2
        PPB_DIM = 250  # Points-per-box-dimension; tuning; higher is slower and more accurate
        RND_SEED = 4    # Random seed; ought to expose for reproducibility

        num_box = np.maximum(int(np.round(multi.area/AREA_TARG)), 1)
        pts_dim = int(np.ceil(np.sqrt(PPB_DIM*num_box)))

        # If the multi polygoin isn't valid; need to bail
        if not multi.is_valid:
            print(k1, 'Multipolygon not valid')
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
        mp = prep(multi)
        points = [Point(t[0], t[1]) for t in pts_vec]
        pts_vec_in = [[p.x, p.y] for p in points if mp.contains(p)]

        # Feed points interior to shape into k-means clustering to get num_box equal(-ish) clusters;
        sub_clust = KMeans(n_clusters=num_box, random_state=RND_SEED, n_init='auto').fit(pts_vec_in)
        sub_node = sub_clust.cluster_centers_
#-------------------
        # Don't actually want the cluster centers, goal is the outlines. Going from centers
        # to outlines uses Voronoi tessellation. Add a box of external points to avoid mucking
        # up the edges. (+/- 200 was arbitrary value greater than any possible lat/long)
        EXT_PTS    = np.array([[-200,-200],[ 200,-200],[-200, 200],[ 200, 200]])
        vor_node  = np.append(sub_node,EXT_PTS,axis=0)
        vor_obj   = Voronoi(vor_node)

        # Extract the Voronoi region boundaries from the Voronoi object. Need to duplicate
        # first point in each region so last == first for the next step
        vor_list = list()
        vor_vert = vor_obj.vertices
        for k2 in range(len(vor_obj.regions)):
            vor_reg = vor_obj.regions[k2]
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
        for k2 in range(len(vor_list)):
            # Voronoi region are convex, so will not need MultiPolygon object
            poly_reg = (Polygon(vor_list[k2])).intersection(multi)

            # Each Voronoi region will be a new shape; give it a name
            new_recs = [rec for rec in sf1r[k1]]  # List copy
            dotname_new = dotname + ':A{:04d}'.format(k2)
            new_recs[dotname_index]  = dotname_new

            # Intersection may be multipolygon; create a poly_as_list representation
            # which will become shapefile
            # poly_as_list = [ [pos_part_1],
            #                  [neg_part_1A],
            #                  [neg_part_1B],
            #                   ... ,
            #                  [pos_part_2],
            #                  [neg_part_2A],
            #                   ... , ...]
            poly_as_list = list()

            if(poly_reg.geom_type == 'MultiPolygon'):

              # Copy/paste from below; multipolygon is just a list of polygons
              for poly_sing in poly_reg.geoms:
                xyset   = poly_sing.exterior.coords
                shp_prt = np.array([[val[0],val[1]] for val in xyset])
                poly_as_list.append(shp_prt.tolist())

                if(len(poly_sing.interiors) > 0):
                  for poly_ing_int in poly_sing.interiors:
                    xyset   = poly_ing_int.coords
                    shp_prt = np.array([[val[0],val[1]] for val in xyset])
                    poly_as_list.append(shp_prt.tolist())

            else:
              xyset   = poly_reg.exterior.coords
              shp_prt = np.array([[val[0],val[1]] for val in xyset])
              poly_as_list.append(shp_prt.tolist())

              if(len(poly_reg.interiors) > 0):
                for poly_ing_int in poly_reg.interiors:
                  xyset   = poly_ing_int.coords
                  shp_prt = np.array([[val[0],val[1]] for val in xyset])
                  poly_as_list.append(shp_prt.tolist())

        # Add the new shape to the shapefile; splat the record
        sf1new.poly(poly_as_list)
        sf1new.record(*new_recs)


# import fiona
# from shapely.geometry import shape
# geoms1 = [shape(pol['geometry']) for pol in fiona.open('datasets/cod_lev02_zones/COD_LEV02_ZONES.shp')]
# geoms = [MultiPolygon([g]) if isinstance(g, Polygon) else g for g in geoms1]
