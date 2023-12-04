from som_methods import som_class
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #####################
    # Load Data
    #####################
    gdf = gpd.read_file("examples/SOM/example_data/lga_pop_landcover.shp")
    gdf = gdf.drop(
        columns=["lga_code", "id", "state_code", "lga_name", "q50", "q025", "q975"]
    )
    feature_names = [
        "log_pop_de",
        "landcover_",
        "landcove_1",
        "landcove_2",
        "landcove_3",
        "landcove_4",
        "landcove_5",
        "landcove_6",
        "landcove_7",
        "landcove_8",
        "landcove_9",
        "landcove10",
        "landcove11",
        "landcove12",
        "landcove13",
        "landcove14",
        "landcove15",
        "landcove16",
        "landcove17",
        "landcove18",
        "landcove19",
        "landcove20",
        "landcove21",
        "landcove22",
        "landcove23",
    ]

    #####################
    # Train SOM
    #####################

    my_som = som_class(gdf, (10, 10), feature_names)
    print(my_som.dimensions)
    my_som.train_som()
    # my_som.plot_som_weights()
    my_som.som_kmeans_clustering(k=3)
    my_som.features_df.plot("cluster")

    titles = [
        "population",
        "Post-flooding or irrigated croplands",
        "Rainfed croplands",
        "Mosaic cropland",
        "Mosaic vegetation",
        "Mosaic forest (50-70%)",
        "Closed to open (>15%) broadleaved evergreen or semi-deciduous forest (>5m)",
        "Closed (>40%) broadleaved evergreen and/or semi-deciduous forest (>5m)",
        "Open (15-40%) broadleaved deciduous forest/woodland (>5m)",
        "Mosaic forest or shrubland (50-70%) / grassland",
        "Mosaic grassland (50-70%) / forest or shrubland (20-50%)",
        "Closed to open (>15%) (broadleaved or needleleaved, evergreen or deciduous) shrubland (<5m)",
        "Closed to open (>15%) broadleaved deciduous shrubland (<5m)",
        "Closed to open (>15%) herbaceous vegetation (grassland, savannas or lichens/mosses)",
        "Closed (>40%) grassland",
        "143-unknown",
        "Sparse (<15%) vegetation",
        "Closed to open (>15%) broadleaved forest regularly flooded (semi-permanently or temporarily) - Fresh or brackish water",
        "Closed (>40%) broadleaved forest or shrubland permanently flooded - Saline or brackish water",
        "Closed to open (>15%) grassland or woody vegetation on regularly flooded or waterlogged soil - Fresh, brackish or saline water",
        "Artificial surfaces and associated areas (Urban areas >50%)",
        "Bare areas",
        "Consolidated bare areas (hardpans, gravels, bare rock, stones, boulders)",
        "Non-consolidated bare areas (sandy desert)",
        "Water bodies",
    ]
    my_som.som_feature_heatmaps(titles=titles)

    # som, df, normalized_features = train_som(["log_pop_density"], projected_gdf, (3,3))
    # print(som.get_weights())

    # fig, axs = plot_som_weights(["log_pop_density"], df, som, (3,3))
    plt.show()
