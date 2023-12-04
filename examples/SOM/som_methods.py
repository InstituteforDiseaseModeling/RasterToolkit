import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from minisom import MiniSom
from typing import List, Tuple


class som_class:
    def __init__(self, features_df, dimensions: tuple, feature_names: list):
        self.features_df = features_df
        self.dimensions = dimensions
        self.feature_names = feature_names

    def train_som(
        self, standardize=False, sigma=1.0, learning_rate=0.5, iterations=100
    ):
        n_features = len(self.feature_names)

        features = self.features_df[self.feature_names].values.astype(np.float64)
        print(f"Features shape: {features.shape}")

        # Initialize and train SOM model
        print(f"Training SOM of shape {self.dimensions}")
        x, y = self.dimensions

        som = MiniSom(
            x, y, input_len=n_features, sigma=sigma, learning_rate=learning_rate
        )
        som.train(features, num_iteration=iterations)

        self.trained_som = som
        print("Finished training SOM.")

        return som

    def som_feature_heatmaps(self, titles=None):
        plt.rcParams["font.size"] = 8

        n_features = len(self.feature_names)

        if titles == None:
            titles = self.feature_names

        som_weights = self.trained_som.get_weights()

        fig, axs = plt.subplots(n_features // 2 + 1, 2, figsize=(14, 56))
        axs = axs.flatten()  # Flatten the array
        fig.suptitle("SOM Weights By Feature")

        for i in range(n_features):
            axs[i].imshow(som_weights[:, :, i], cmap="coolwarm", interpolation="none")
            # axs[i].set_title(titles[i][:50])
            # fig.colorbar(
            #     plt.cm.ScalarMappable(cmap="coolwarm"), ax=axs[i], label="weights"
            # )

        # Show the plot
        for ax in axs[n_features:]:
            ax.axis("off")
        # fig.tight_layout(rect=[0, 0.03, 1, 0.99])
        plt.tight_layout()
        plt.show()

        # return fig, axs

    def som_kmeans_clustering(self, k=10):
        n_features = len(self.feature_names)
        x, y = self.dimensions

        som_weights = self.trained_som.get_weights()
        weights_reshaped = np.reshape(som_weights, (x * y, n_features))

        kmeans = KMeans(n_clusters=k, random_state=0).fit(weights_reshaped)
        self.kmeans_clusters = kmeans

        self.get_cluster_membership(clustering_type="kmeans")

        return kmeans

    def coordinates_to_cluster_label(self, coordinates):
        x, y = coordinates
        return x * 3 + y + 1

    def get_winner_coordinates(self, data_point):
        x, y = self.trained_somsom.winner(data_point)
        return (x, y)

    def get_cluster_membership(self, clustering_type):
        n_features = len(self.feature_names)
        x, y = self.dimensions

        features = self.features_df[self.feature_names].values.astype(np.float64)

        if clustering_type == "kmeans":
            winning_neurons = np.array(
                [self.trained_som.winner(feat) for feat in features]
            )

            # convert winning_neurons to a 1D array of integers for easy comparison with kmeans.labels_
            winning_neurons_1d = np.ravel_multi_index(
                winning_neurons.T, self.dimensions
            )

            # create a dictionary mapping from neuron index to cluster
            neuron_to_cluster = dict(zip(range(x * y), self.kmeans_clusters.labels_))

            # map each row to its corresponding cluster
            self.features_df["cluster"] = [
                neuron_to_cluster[i] for i in winning_neurons_1d
            ]

        elif clustering_type == "som":
            winning_coordinates = [self.get_winner_coordinates(dp) for dp in features]
            self.features_df["cluster"] = [
                self.coordinates_to_cluster_label(coord)
                for coord in winning_coordinates
            ]
