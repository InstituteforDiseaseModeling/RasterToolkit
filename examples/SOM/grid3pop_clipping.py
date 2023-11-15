# import geopandas as gpd
import rasterio
import pandas as pd
from pathlib import Path
from typing import Union
# import time
from rastertools import raster_clip


if __name__ == "__main__":
    print("running")
    # start = time.time()
    raster_file_path = "example_data/NGA_population_v2_0_gridded/NGA_population_v2_0_gridded.tif"
    shape_file_path = "example_data/geonetwork_landcover_nga_gc_adg/nga_gc_adg.shp"

    pop_dict = raster_clip(raster_file_path, shape_file_path, include_latlon=True)
    df = pd.DataFrame.from_dict(pop_dict, orient="index")
    df = df.reset_index(names=["dot_name"])
    df.to_csv("results/clipped_grid3_pop.csv", index=False)



