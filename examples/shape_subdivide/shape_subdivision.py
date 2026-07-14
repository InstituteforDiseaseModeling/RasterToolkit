"""
Example showing how to use shape subdivision API to split shapes into Voronoi
sub-shapes.
"""

import matplotlib.pyplot as plt
import os
import time

from rastertoolkit import shape_subdivide
from rastertoolkit.shape import plot_shapes


def subdivide_example(area: int = None):

    shape_file = os.path.join("..", "data", "COD_LEV02_ZONES")

    time_start = time.time()
    print(f"Starting {area or 'default'} subdivision...")

    new_shape_stem = shape_subdivide(shape_stem=shape_file,
                                     out_dir=".",
                                     box_target_area_km2=area,
                                     verbose=True)

    time_end = time.time()
    print(f"Completed subdivision in {round(time_end - time_start)}s")

    print("Plotting admin shapes and new subdivision layer.")
    fig = plt.figure()
    axs = fig.add_subplot(1, 1, 1, label=None)

    plot_shapes(shape_file,
                ax=axs,
                alpha=0.5,
                color=None,
                linewidth=1.0,
                edgecolor="gray")

    plot_shapes(new_shape_stem,
                ax=axs,
                alpha=0.3,
                color=None,
                linewidth=0.2,
                edgecolor="red")

    fig.savefig(new_shape_stem + ".png", dpi=600)


subdivide_example()  # default is 100 km2
subdivide_example(400)

print("Finished processing.")
