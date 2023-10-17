#!/usr/bin/env python
"""
Example showing how to use rastertools API to population data from WorldPop raster using shapes and selectors.
"""

import os
import pandas as pd

from pathlib import Path
from rastertools import download, raster_clip, utils
from typing import Dict

# GDX Download
os.environ['CKAN_API_KEY'] = Path("gdx.key").read_text()          # GDx API KEY
shp = download("23930ae4-cd30-41b8-b33d-278a09683bac", extract=True)    # DRC health zones shapefiles
rst = download("0c7241d0-a31f-451f-9df9-f7f3eff81e03", extract=True)    # WorldPop rasters - DRC, 1km, UN adj
shape_file = Path(shp[0])
shape_file = shape_file.parent.joinpath(shape_file.stem)
raster_file = Path(rst[-1])

# Clipping raster with shapes (only pop values)
pop_dict: Dict = raster_clip(raster_file, shape_file)

# Save to a local file json
utils.save_json(pop_dict, json_path="results/clipped_pop.json", sort_keys=True)

# Clipping raster with shapes (including lat/lon)
pop_dict2: Dict = raster_clip(raster_file, shape_file, include_latlon=True)

# Save to a local csv file (include lat/lon)
df = pd.DataFrame.from_dict(pop_dict2, orient="index")
df = df.reset_index(names=["dot_name"])
df.to_csv("results/clipped_pop.csv", index=False)

# To generate EMOD demographics you can use emod_api Demographics object
# dmg: Demographics = Demographics.from_csv(pop_csv)


