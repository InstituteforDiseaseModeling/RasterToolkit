"""
Example showing how to use shape subdivision API to split shapes into Voronoi
sub-shapes.
"""

import os
import time

from rastertoolkit import shape_subdivide
from rastertoolkit.shape import plot_subdivision


def subdivide_example(area: int = None):

    shape_file = os.path.join("..", "data", "COD_LEV02_ZONES")

    start_time = time.time()

    print(f"Starting {area or 'default'} subdivision...")
    new_shape_stem = shape_subdivide(shape_stem=shape_file, out_dir=".",
                                     box_target_area_km2=area, verbose=True)
    print(f"Completed subdivision in {round(time.time() - start_time)}s")

    print("Plotting admin shapes and new subdivision layer.")
    plot_subdivision(shape_file, new_shape_stem)


subdivide_example()  # default is 100 km2
subdivide_example(400)

print("Finished processing.")
