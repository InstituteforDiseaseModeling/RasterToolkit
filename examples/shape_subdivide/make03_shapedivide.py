#*******************************************************************************
import os, shapefile

import numpy              as np

import matplotlib.path    as mplp

from shapely.geometry import Polygon, MultiPolygon
from sklearn.cluster  import KMeans
from scipy.spatial    import Voronoi

from rastertools  import  download, area_sphere, get_remote
from rastertools.shape  import *

#*******************************************************************************
# Quick, approximate  scaling for longitute ratio as a function of latitude
def long_mult(lat): # latitude in degrees

  return 1.0/np.cos(lat*np.pi/180.0)

#*******************************************************************************


DATA_ROOT = "datasets" # os.path.join(os.getenv('DATA_ROOT', "dataset"),'GDx')
TLC       = 'COD'


# GDX - Download DRC health zones shapefiles
shp = download(data_id  = '23930ae4-cd30-41b8-b33d-278a09683bac',
               data_dir = DATA_ROOT,
               extract  = True,
               remote=get_remote("../../gdx.key"))

file_name   = '{:s}_LEV02_ZONES'.format(TLC)
shape_path  = os.path.join(DATA_ROOT, '{:s}_lev02_zones'.format(TLC.lower()), file_name)


# Input shapefiles
sf1    = shapefile.Reader(shape_path)
sf1s   = sf1.shapes()
sf1r   = sf1.records()


# Output shapefile
sf1new = shapefile.Writer(file_name + '_100km')

# Output shapefile should get all the same fields as input shapefile. First field
# in the list (index zero) is always an auto-added deletion flag for database purposes,
# so we skip it. It's not present in the records entry, so there's an off-by-one thing
# going on (by design). I also add 10 characters to the length of the 'DOTNAME' field.
# Unfortunately 'DOTNAME' is not a standard label, but I use it like it is. Each field has
# three parts ['<LABEL>', '<DATATYPE>', <LENGTH>]; so something like ['DOTNAME','C',60]

dotname_index = None
for k1 in range(1,len(sf1.fields)):
  if(sf1.fields[k1][0] == 'DOTNAME'):
    dotname_index = (k1-1)
    sf1new.field(sf1.fields[k1][0],sf1.fields[k1][1],sf1.fields[k1][2]+10)
  else:
    sf1new.field(sf1.fields[k1][0],sf1.fields[k1][1],sf1.fields[k1][2])

# DEJAN - MultiPolygon
multi_polygons = shapes_to_polygons(sf1)

# Iterate over every shape in the shapefile
for k1 in range(len(sf1r)):
  dotname       = sf1r[k1][dotname_index]
  wrk_shape     = sf1s[k1]
  wrk_shape_pts = np.array(sf1s[k1].points)
  prt_list      = list(wrk_shape.parts) + [len(wrk_shape_pts)]

  # First step is converting this shape from the raw shapefile representation
  # into an instance of a shapely MultiPolygon. Calculating intersections between
  # complicated shapes is rough, so easiest to use shapely.

  # Raw shape file data is annoying. One shape in a shape file could be something
  # like two non-overlapping donuts (or two non overlapping slices of swiss cheese
  # since there can be many holes). The ESRI standard is the each positive part is
  # immediately followed by 0 or more negative holes in the positive part. In the
  # constructed shp_list below, everything except pos_part_1 is optional.
  #
  # shp_list = [ [pos_part_1, neg_part_1A, neg_part_1B, ...],
  #              [pos_part_2, neg_part_2A, neg_part_2B, ...], ...]

  # Build shp_list from raw shape data (retain both tot_area and shp_list)
  shp_list = list()
  tot_area = 0
  for k2 in range(len(prt_list)-1):
    shp_prt   = wrk_shape_pts[prt_list[k2]:prt_list[k2+1],:]
    prt_area  = area_sphere(shp_prt)
    tot_area  = tot_area + prt_area
    if(prt_area > 0):
      shp_list.append([shp_prt])
    else:
      shp_list[-1].append(shp_prt)

  # Given shp_list, construct MultiPolygon (retain mltigon)
  ply_list = list()
  for shp_prt in shp_list:
    if(len(shp_prt) == 1):
      ply_list.append(Polygon(shp_prt[0]))
    else:
      ply_list.append(Polygon(shp_prt[0], holes=shp_prt[1:]))
  mltigon = MultiPolygon(ply_list)

  # DEJAN - compare multi-polygons flavors
  mltign2 = multi_polygons[k1]   # --- DEJAN ---
  assert mltigon == mltign2, "multi-polygons must be the same"


  # Second step is to create an underlying mesh of points. If the mesh is
  # equidistant, then the subdivided shapes will be uniform area. Alternatively,
  # the points could be population raster data, and the subdivided shapes would
  # be uniform population.

  AREA_TARG  = 100  # Needs to be configurable; here target is ~100km^2
  PPB_DIM    = 250  # Points-per-box-dimension; tuning; higher is slower and more accurate
  RND_SEED   = 4    # Random seed; ought to expose for reproducability

  # DEJAN - area diff
  tot_area2 = polygon_area_km2(mltign2)
  area_diff = abs(tot_area - tot_area2)/tot_area
  assert area_diff < 0.01

  num_box    = np.maximum(int(np.round(tot_area/AREA_TARG)),1)
  pts_dim    = int(np.ceil(np.sqrt(PPB_DIM*num_box)))


  # If the multipolygoin isn't valid; need to bail
  if not mltigon.is_valid:
    print(k1, 'Multipolygon not valid')
    continue

  # Debug logging: shapefile index, target number of subdivisions
  print(k1, num_box)

  # Start with a rectangular mesh, then (roughly) correct longitude (x values);
  # Assume spacing on latitude (y values) is constant; x value spacing needs to
  # be increased based on y value.

  xspan = [np.min(wrk_shape_pts[:,0]),np.max(wrk_shape_pts[:,0])]
  yspan = [np.min(wrk_shape_pts[:,1]),np.max(wrk_shape_pts[:,1])]
  (xcv,ycv) = np.meshgrid(np.linspace(xspan[0],xspan[1],pts_dim),
                          np.linspace(yspan[0],yspan[1],pts_dim))

  pts_vec      = np.zeros((pts_dim*pts_dim,2))
  pts_vec[:,0] = np.reshape(xcv,(pts_dim*pts_dim))
  pts_vec[:,1] = np.reshape(ycv,(pts_dim*pts_dim))
  pts_vec[:,0] = pts_vec[:,0]*long_mult(pts_vec[:,1]) - xspan[0]*(long_mult(pts_vec[:,1])-1)
  inBool       = np.zeros(pts_vec.shape[0],dtype=bool)


  # Same idea here as in raster clipping; identify points that are inside the shape
  # and keep track of them using inBool
  for k2 in range(len(shp_list)):
    for k3 in range(len(shp_list[k2])):
      path_shp  = mplp.Path(shp_list[k2][k3],closed=True,readonly=True)
      if(k3 == 0):
        part_bool = path_shp.contains_points(pts_vec)
      else:
        part_bool = np.logical_and(part_bool,np.logical_not(path_shp.contains_points(pts_vec)))
    inBool = np.logical_or(inBool,part_bool)

  # DEJAN - contains points
  pts_vec_in1: np.ndarray = pts_vec[inBool, :]
  pts_vec_in2: np.ndarray = polygon_contains(mltign2, pts_vec)

  len1, len2 = pts_vec_in1.shape[0], pts_vec_in2.shape[0]
  max_i = min(len1, len2)
  pts_vec_diff = [t for t in  enumerate(pts_vec_in1[:max_i,:] - pts_vec_in2[:max_i,]) if sum(t[1]) != 0]
  if len1 != len2 or len(pts_vec_diff) > 0:
    last_few1 = pts_vec_in1[-3:, :]
    last_few2 = pts_vec_in2[-3:, :]
    print("points vectors differ")
  # try:
  #   assert np.array_equal(pts_vec_in1, pts_vec_in2), "Containing points must be the same"
  #   #assert (pts_vec_in1 == pts_vec_in2).all(), "Containing points must be the same"
  # except Exception as ex:
  #
  #   pass

  # Feed points interior to shape into k-means clustering to get num_box equal(-ish) clusters;
  sub_clust = KMeans(n_clusters=num_box, random_state=RND_SEED, n_init='auto').fit(pts_vec[inBool,:])
  sub_node  = sub_clust.cluster_centers_


  # Don't actually want the cluster centers, goal is the outlines. Going from centers
  # to outlines uses Voronoi tessellation. Add a box of external points to avoid mucking
  # up the edges. (+/- 200 was arbitrary value greater than any possible lat/long)
  EXT_PTS    = np.array([[-200,-200],[ 200,-200],[-200, 200],[ 200, 200]])
  vor_node  = np.append(sub_node,EXT_PTS,axis=0)
  vor_obj   = Voronoi(vor_node)


  # Extract the Voronoi region boundaries from the Voronoi object. Need to duplicate
  # first point in each region so last == first for the next step
  vor_list  = list()
  vor_vert  = vor_obj.vertices
  for k2 in range(len(vor_obj.regions)):
    vor_reg = vor_obj.regions[k2]
    if(-1 in vor_reg or len(vor_reg) == 0):
      continue
    vor_loop = np.append(vor_vert[vor_reg,:],vor_vert[vor_reg[0:1],:],axis=0)
    vor_list.append(vor_loop)


  # If there's not 1 Voronoi region outline for each k-means cluster center
  # at this point, something has gone very wrong. Time to bail.
  if(len(vor_list) != len(sub_node)):
    print(k1, 'BLARG')
    1/0


  # The Voronoi region outlines may extend beyond the shape outline and/or
  # overlap with negative spaces, so intersect each Voronoi region with the
  # shapely MultiPolygon created previously
  for k2 in range(len(vor_list)):

    # Voronoi region are convex, so will not need MultiPolygon object
    poly_reg  = (Polygon(vor_list[k2])).intersection(mltigon)

    # Each Voronoi region will be a new shape; give it a name
    new_recs                 = [rec for rec in sf1r[k1]]  # List copy
    dotname_new              = dotname + ':A{:04d}'.format(k2)
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

sf1new.close()

