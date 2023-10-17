#!/usr/bin/env python
"""
Example showing how to use shape subdivision API to split shapes into Voronoi sub-shapes.
"""

import os
import time

from rastertools import download, shape_subdivide
from rastertools.shape import plot_subdivision

from pathlib import Path
from typing import List

# GDX Download
os.environ['CKAN_API_KEY'] = Path("gdx.key").read_text()                       # GDx API KEY
shp: List[str] = download("23930ae4-cd30-41b8-b33d-278a09683bac", extract=True)  # DRC health zones shapefiles

# Shape file paths
shape_file = Path(shp[0])  # Shape file path
out_dir = "results"        # output dir


def subdivide_example(area: int = None):
    start_time = time.time()

    print(f"Starting {area or 'default'} subdivision...")
    new_shape_stam = shape_subdivide(shape_stem=shape_file, out_dir=out_dir, box_target_area_km2=area, verbose=True)
    print(f"Completed subdivision in {round(time.time() - start_time)}s")

    print(f"Plotting admin shapes and new subdivision layer.")
    plot_subdivision(shape_file, new_shape_stam)
    print(f"Completed plotting.")


subdivide_example()  # default is 100 km2
subdivide_example(400)

print(f"Finished processing.")


