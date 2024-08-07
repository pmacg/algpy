"""Implementation of the Dataset object for use with algpy."""
from sklearn.datasets import make_moons
import numpy as np
from abc import ABC, abstractmethod
import stag.cluster
import matplotlib.pyplot as plt

class Dataset(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """Construct the dataset."""
        pass


class PointCloudDataset(Dataset):
    """
    The simplest form of dataset: the data consists of a point cloud in Euclidean space.
    This is represented internally by a numpy array.
    """

    def __init__(self, data: np.array=None, labels=None):
        """Initialise the dataset with a numpy array. Optionally, provide labels for classification."""
        self.data = np.array(data)
        self.n, self.d = data.shape
        self.gt_labels = labels

    def plot_clusters(self, labels):
        """
        If the data is two-dimensional, plot the data, colored according to the labels.
        """
        if self.d != 2:
            raise ValueError("Dataset is not two-dimensional")

        labels = np.array(labels)

        # Plot the data points, colored by their cluster labels
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_data = self.data[labels == label]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}')

        plt.grid(True)
        plt.show()


class TwoMoonsDataset(PointCloudDataset):
    """The toy two moons dataset from sklearn."""

    def __init__(self, n=1000, noise=0.07):
        """Initialise the two moons dataset. Optionally, provide the number of points, n, and the noise parameter."""
        X, y = make_moons(n_samples=n, noise=noise)
        super(TwoMoonsDataset, self).__init__(data=X, labels=y)

    def __str__(self):
        return f"TwoMoonsDataset({self.n})"
