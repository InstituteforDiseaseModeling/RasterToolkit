#!/usr/bin/env python
"""
Example showing how to use shape subdivision API to split shapes into Voronoi sub-shapes.
"""

import matplotlib.path as mplp
import numpy as np
import os
import shapefile
import time

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon
from sklearn.cluster import KMeans

from pathlib import Path
from rastertools import download, shape_subdivide


# GDX Download
os.environ['CKAN_API_KEY'] = Path("../../gdx.key").read_text()          # GDx API KEY
shp = download("23930ae4-cd30-41b8-b33d-278a09683bac", extract=True)    # DRC health zones shapefiles

# Shape file paths
shape_file = Path(shp[0])
shape_file = shape_file.parent.joinpath(shape_file.stem)
new_shape_file = f"results2/actual/{shape_file.stem}_100km"

# Processing
start_time = time.time()
sub_shapes = shape_subdivide(shape_file, out_shape_stem=new_shape_file)
dt = time.time() - start_time
print(f"--- {int(dt//60)}m {int(dt%60)}s ---")

