from minisom import MiniSom
from pathlib import Path
from rastertools import download, utils

import numpy as np
import rasterio
import os

os.environ['CKAN_API_KEY'] = Path("gdx.key").read_text()
rst = download("0c7241d0-a31f-451f-9df9-f7f3eff81e03", extract=True)
raster_file = Path(rst[-1])

with rasterio.open(raster_file) as src:
    data = src.read(1)

data = data.reshape((-1, 1))


test = 1