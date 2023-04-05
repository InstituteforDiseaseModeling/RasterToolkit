#!/usr/bin/env python
"""
Example showing how to use shape subdivision API to split shapes into Voronoi sub-shapes.
"""

import os
import time

import rastertools as rst

from pathlib import Path
from typing import List

# GDX Download
os.environ['CKAN_API_KEY'] = Path("../../gdx.key").read_text()          # GDx API KEY
shp: List[str] = rst.download("23930ae4-cd30-41b8-b33d-278a09683bac", extract=True)    # DRC health zones shapefiles

# Shape file paths
shape_file = Path(shp[0])
shape_file = shape_file.parent.joinpath(shape_file.stem)
new_shape_file = f"results/actual/{shape_file.stem}_100km"

# Processing
start_time = time.time()
rst.shape_subdivide(shape_stem=shape_file, out_shape_stem=new_shape_file)  #, top_n=3)
dt = time.time() - start_time
print(f"Subdivision completed in {int(dt//60)}m {int(dt%60)}s")

# Plot generated shapes into a file
png_file = Path(new_shape_file).with_suffix(".png")
fig, ax = rst.shape.plot_shapes(shape_file, color="gray", alpha=0.5, line_width=1.0)
rst.shape.plot_shapes(new_shape_file, ax=ax, color="red", alpha=0.3, line_width=0.2)
fig.savefig(png_file, dpi=1800)

print(f"Finished processing.")


