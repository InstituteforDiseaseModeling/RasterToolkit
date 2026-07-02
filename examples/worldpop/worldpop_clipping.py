"""
Example showing how to use rastertoolkit API to population data from WorldPop
raster using shapes and selectors.
"""

import csv
import json
import os

from rastertoolkit import raster_clip

# Using example DRC shapefile and raster
shape_file = os.path.join("..", "data", "COD_LEV02_ZONES")
raster_file = os.path.join("..", "data", "cod_2020_1km_aggregated_unadj.tif")

# Clipping raster with shapes (only pop values)
popdict1 = raster_clip(raster_file, shape_file)

# Save to a local file json
with open("clipped_pop.json", "w") as fid01:
    json.dump(popdict1, fid01, sort_keys=True, indent=4)

# Clipping raster with shapes (including lat/lon)
popdict2 = raster_clip(raster_file, shape_file, include_latlon=True)

# Save to a local csv file (include lat/lon)
with open("clipped_pop.csv", "w", newline='') as csvfile:
    fieldnames = ['NAME', 'LAT', 'LON', 'POP']
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()
    for shapekey in popdict2:
        tmp_dict = {'NAME': shapekey,
                    'LAT': popdict2[shapekey]['lat'],
                    'LON': popdict2[shapekey]['lon'],
                    'POP': popdict2[shapekey]['pop']}
        csvwriter.writerow(tmp_dict)
