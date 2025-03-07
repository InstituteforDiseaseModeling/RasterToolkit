#!/usr/bin/env python
"""
Example showing how to use rastertoolkit API to population data from WorldPop
raster using shapes and selectors.
"""

import csv

from pathlib import Path
from rastertoolkit import raster_clip, utils
from typing import Dict

# Using example DRC shapefile and raster
shape_file = Path('../data/COD_LEV02_ZONES')
raster_file = Path('../data/cod_2020_1km_aggregated_unadj.tif')

# Clipping raster with shapes (only pop values)
popdict1: Dict = raster_clip(raster_file, shape_file)

# Save to a local file json
utils.save_json(popdict1, json_path="results/clipped_pop.json", sort_keys=True)

# Clipping raster with shapes (including lat/lon)
popdict2: Dict = raster_clip(raster_file, shape_file, include_latlon=True)

# Save to a local csv file (include lat/lon)
with open("results/clipped_pop.csv", 'w', newline='') as csvfile:
    fieldnames = ['NAME', 'LAT', 'LON', 'POP']
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()
    for shapekey in popdict2:
        tmp_dict = {'NAME': shapekey,
                    'LAT': popdict2[shapekey]['lat'],
                    'LON': popdict2[shapekey]['lon'],
                    'POP': popdict2[shapekey]['pop']}
        csvwriter.writerow(tmp_dict)
