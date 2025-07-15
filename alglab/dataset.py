"""Implementation of the Dataset object for use with algpy."""
from sklearn.datasets import make_moons, fetch_openml, make_blobs, make_circles
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from abc import ABC, abstractmethod
import stag.graph
import stag.random
import matplotlib.pyplot as plt
from typing import Type, Dict, List, Tuple
import pandas as pd


class Dataset(ABC):

    def __init__(self, **kwargs):
        """Construct the dataset."""
        self.num_steps = 1

    def get_step(self, step_id):
        """Get the update which must happen at the given step.
        An update has the form (method_name, arguments, gt_output)."""
        return


class NoDataset(Dataset):
    """Use this when no dataset is needed to compare algorithms."""

    def __init__(self):
        super().__init__()
        self.num_steps = 0

    def __str__(self):
        return "NoDataset"


class ClusterableDataset(Dataset):
    """
    A dataset which may have ground truth clusters.
    """

    def __init__(self, labels, **kwargs):
        super().__init__(**kwargs)
        self.gt_labels = labels

    def cluster_ids(self):
        if self.gt_labels is None:
            return None
        else:
            return np.unique(self.gt_labels).tolist()

    def get_cluster(self, cluster_id: int):
        cluster = []
        for id, lab in enumerate(self.gt_labels):
            if lab == cluster_id:
                cluster.append(id)
        return cluster

    def get_step(self, step_id):
        return 'fit_predict', None, self.gt_labels


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

    def get_step(self, step_id):
        return 'fit_predict', self.graph, self.gt_labels


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
        self.k = -1 if labels is None else len(np.unique(np.asarray(labels)))
        ClusterableDataset.__init__(self, labels)

    def apply_pca(self, new_dimension: int):
        pca = PCA(n_components=new_dimension)
        self.data = pca.fit_transform(self.data)
        assert self.data.shape[1] == new_dimension
        self.d = new_dimension

    def apply_scaling(self):
        scaler = StandardScaler().fit(self.data)
        self.data = scaler.transform(self.data)

    def plot_clusters(self, labels, dimension_idxs=None):
        """
        If the data is two-dimensional, plot the data, colored according to the labels.

        If dimension_idxs is an array of length two, plot the data using the given dimension indices.
        """
        if (self.d != 2 and dimension_idxs is None) or (dimension_idxs is not None and len(dimension_idxs) != 2):
            raise ValueError("Cannot plot dataset: it has more than two dimensions.")

        if len(labels) != self.n:
            raise ValueError("Cannot plot dataset: labels length must match number of data points.")

        if dimension_idxs is None:
            dimension_idxs = [0, 1]

        labels = np.array(labels)

        # Plot the data points, colored by their cluster labels
        plt.figure(figsize=(10, 7))
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_data = self.data[labels == label]
            plt.scatter(cluster_data[:, dimension_idxs[0]], cluster_data[:, dimension_idxs[1]], label=f'Cluster {label}')

        plt.grid(True)
        plt.show()

    def plot_data(self, dimension_idxs=None):
        """
        If the data is two-dimensional, plot it.

        If dimension_idxs is an array of length two, plot the data using the given dimension indices.
        """
        labels = np.ones(self.n)
        self.plot_clusters(labels, dimension_idxs=dimension_idxs)

    def get_step(self, step_id):
        return 'fit_predict', self.data, self.gt_labels


class TwoMoonsDataset(PointCloudDataset):
    """The toy two moons dataset from sklearn."""

    def __init__(self, n=1000, noise=0.07):
        """Initialise the two moons dataset. Optionally, provide the number of points, n, and the noise parameter."""
        x, y = make_moons(n_samples=int(n), noise=noise)
        PointCloudDataset.__init__(self, data=x, labels=y)

    def __str__(self):
        return f"TwoMoonsDataset({self.n})"


class BlobsDataset(PointCloudDataset):
    """The toy blobs dataset from sklearn."""

    def __init__(self, n=1000, d=2, k=3):
        """Initialise the blobs dataset. Optionally, provide the number of points, dimensions, and clusters."""
        x, y = make_blobs(n_samples=n, n_features=d, centers=k)
        PointCloudDataset.__init__(self, data=x, labels=y)

    def __str__(self):
        return f"BlobsDataset({self.n}, {self.d}, {self.k})"


class CirclesDataset(PointCloudDataset):
    """The toy circles dataset from sklearn."""

    def __init__(self, n=1000, noise=0.07):
        """Initialise the circles dataset. Optionally, provide the number of points and the noise parameter."""
        x, y = make_circles(n_samples=int(n), noise=noise)
        PointCloudDataset.__init__(self, data=x, labels=y)

    def __str__(self):
        return f"CirclesDataset({self.n})"


class OpenMLDataset(PointCloudDataset):
    """Load pointcloud data from OpenML."""

    def __init__(self, **kwargs):
        """Initialise the dataset by downloading from openML. Accepts the same arguments as the
        sklearn fetch_openml method."""
        data_info = fetch_openml(**kwargs)
        if isinstance(data_info.data, pd.DataFrame):
            data_info.data = data_info.data.to_numpy()

        target = data_info.target
        if isinstance(target, pd.Series) or isinstance(target, pd.DataFrame):
            if isinstance(target.dtype, pd.CategoricalDtype):
                target = target.cat.codes.to_numpy()
            else:
                target = data_info.target.to_numpy()

        PointCloudDataset.__init__(self, data=data_info.data, labels=target)


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

