# import geopandas as gpd
import rasterio
import pandas as pd
from pathlib import Path
from typing import Union

# import time
# from rastertools import raster_clip
from som_methods import som_class
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #####################
    # Load Data
    #####################

    projected_gdf = gpd.read_file("examples/SOM/example_data/GRID3_NGA_population_v2_0_admin_Ward_processed.shp")
    print(projected_gdf.columns)

    #####################
    # Plot Population Variables
    #####################
    plt.rc("font", size=14)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Grid3 Nigeria Population Estimates at Ward Level")
    projected_gdf.plot(
        "mean",
        ax=axs[0],
        legend=True,
        legend_kwds={"location": "bottom", "shrink": 0.75},
    )
    axs[0].set_title("Population")
    axs[0].axis("off")

    projected_gdf.plot(
        "log_pop_de",
        ax=axs[1],
        legend=True,
        legend_kwds={"location": "bottom", "shrink": 0.75},
    )
    axs[1].set_title("Log Population density - (population/km^2)")
    axs[1].axis("off")

    #####################
    # Train SOM
    #####################
    my_som = som_class(projected_gdf, (10, 10), ["log_pop_de"])
    print(my_som.dimensions)
    my_som.train_som()
    # my_som.som_feature_heatmaps()
    my_som.som_kmeans_clustering(k=3)

    #####################
    # Plot Cluster Membership
    #####################
    my_som.features_df.plot("cluster")
    plt.title("Cluster Membership")

    plt.show()
