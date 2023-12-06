from pathlib import Path
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
from rastertools import download, shape_subdivide
from rastertools.shape import plot_subdivision
from shapefile import Shape, ShapeRecord, Reader, Shapes, Writer, POINT


# Shape file paths
out_dir = "results" # output dir
shape_file = "example_data/NGA_population_v2_0_admin/GRID3_NGA_population_v2_0_admin_LGA.shp"  # Shape file path

def subdivide_example(area: int = None):
    start_time = time.time()

    print(f"Starting {area or 'default'} subdivision...")
    new_shape_stam = shape_subdivide(shape_stem=shape_file, out_dir=out_dir, box_target_area_km2=area, verbose=True)
    print(f"Completed subdivision in {round(time.time() - start_time)}s")


    print(f"Plotting admin shapes and new subdivision layer.")
    plot_subdivision(shape_file, new_shape_stam)
    print(f"Completed plotting.")


subdivide_example()  # default is 100 km2

print(f"Finished processing.")