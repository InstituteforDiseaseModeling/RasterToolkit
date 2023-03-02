#!/usr/bin/env python
"""
Example showing how to use shape subdivision API to split shapes into Voronoi sub-shapes.
"""

import matplotlib.path as mplp
import numpy as np
import os
import shapefile

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon
from sklearn.cluster import KMeans

from pathlib import Path
from rastertools import download, shape_subdivide


# GDX Download
os.environ['CKAN_API_KEY'] = Path("../../gdx.key").read_text()          # GDx API KEY
shp = download("23930ae4-cd30-41b8-b33d-278a09683bac", extract=True)    # DRC health zones shapefiles

shape_file = Path(shp[0])
shape_file = shape_file.parent.joinpath(shape_file.stem)
new_shape_file = f"results/{shape_file.stem}_100km"

import pickle# Output shapefile
# TLC = 'COD'
# file_name = f"{TLC}_LEV02_ZONES"
# new_shape_file = Path(f"results/{TLC}_lev02_zones/{TLC}_LEV02_ZONES")

sub_shapes = shape_subdivide(shape_file, out_shape_stem=new_shape_file)


