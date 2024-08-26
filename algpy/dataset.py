"""Implementation of the Dataset object for use with algpy."""
from sklearn.datasets import make_moons, fetch_openml
from sklearn.neighbors import kneighbors_graph
import numpy as np
from abc import ABC, abstractmethod
import stag.graph
import stag.random
import matplotlib.pyplot as plt
from typing import Type, Dict
import pandas as pd


class Dataset(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """Construct the dataset."""
        pass


class NoDataset(Dataset):
    """Use this when no dataset is needed to compare algorithms."""

    def __init__(self):
        pass

    def __str__(self):
        return "NoDataset"


class ClusterableDataset(Dataset):
    """
    A dataset which may have ground truth clusters.
    """

    def __init__(self, labels):
        self.gt_labels = labels


class GraphDataset(ClusterableDataset):
    """
    A dataset whose central data is a graph.
    """

    def __init__(self, graph: stag.graph.Graph = None, labels=None):
        """Initialise the dataset with a stag Graph. Optionally, provide ground truth
        labels for classification."""
        self.graph = graph
        self.n = 0 if graph is None else graph.number_of_vertices()
        ClusterableDataset.__init__(self, labels)


class SBMDataset(GraphDataset):
    """
    Create a graph dataset from a stochastic block model.
    """

    def __init__(self, n: int = 1000, k: int = 10, p: float = 0.5, q: float = 0.1):
        self.n = int(n)
        self.k = int(k)
        self.p = p
        self.q = q
        g = stag.random.sbm(self.n, self.k, p, q)
        labels = stag.random.sbm_gt_labels(self.n, self.k)
        GraphDataset.__init__(self, graph=g, labels=labels)


    def __repr__(self):
        return f"SBMDataset({self.n}, {self.k}, {self.p}, {self.q})"


class PointCloudDataset(ClusterableDataset):
    """
    The simplest form of dataset: the data consists of a point cloud in Euclidean space.
    This is represented internally by a numpy array.
    """

    def __init__(self, data: np.array = None, labels=None):
        """Initialise the dataset with a numpy array. Optionally, provide labels for classification."""
        self.data = np.array(data)
        self.n, self.d = data.shape
        ClusterableDataset.__init__(self, labels)

    def plot_clusters(self, labels):
        """
        If the data is two-dimensional, plot the data, colored according to the labels.
        """
        if self.d != 2:
            raise ValueError("Dataset is not two-dimensional.")

        if len(labels) != self.n:
            raise ValueError("Labels length must match number of data points.")

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
        x, y = make_moons(n_samples=n, noise=noise)
        PointCloudDataset.__init__(self, data=x, labels=y)

    def __str__(self):
        return f"TwoMoonsDataset({self.n})"


class OpenMLDataset(PointCloudDataset):
    """Load pointcloud data from OpenML."""

    def __init__(self, **kwargs):
        """Initialise the dataset by downloading from openML. Accepts the same arguments as the
        sklearn fetch_openml method."""
        data_info = fetch_openml(**kwargs)
        if isinstance(data_info.data, pd.DataFrame):
            data_info.data = data_info.data.to_numpy()

        if isinstance(data_info.target, pd.Series) or isinstance(data_info.target, pd.DataFrame):
            data_info.target.to_numpy()

        PointCloudDataset.__init__(self, data=data_info.data, labels=data_info.target)


class KnnGraphDataset(GraphDataset, PointCloudDataset):
    """A k-nearest neighbour graph dataset is both a point cloud and a graph dataset."""

    def __init__(self,
                 k: int = 10,
                 pointcloud_class: Type[PointCloudDataset] = PointCloudDataset,
                 **pointcloud_parameters):
        # Initialise this as a pointcloud dataset
        pointcloud_class.__init__(self, **pointcloud_parameters)

        # Create the k nearest neighbours graph and initialise as a graph dataset
        adj_non_symmetric = kneighbors_graph(self.data, k)
        g = stag.graph.Graph(adj_non_symmetric + adj_non_symmetric.transpose())
        GraphDataset.__init__(self, graph=g, labels=self.gt_labels)
