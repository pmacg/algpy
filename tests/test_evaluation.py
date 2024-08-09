"""Test the evaluation methods of algpy."""
import pytest
import stag.graph
import stag.random
from sklearn.cluster import KMeans
import numpy as np
import algpy.dataset
import algpy.evaluation


# We will use this KMeans implementation throughout the tests.
def kmeans_impl(data: algpy.dataset.PointCloudDataset, k=10):
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def test_ari():
    # Create a dataset with ground truth
    data = algpy.dataset.TwoMoonsDataset()

    best_ari = algpy.evaluation.adjusted_rand_index.apply(data, data.gt_labels)
    assert best_ari == 1

    kmeans_labels = kmeans_impl(data, k=2)
    kmeans_ari = algpy.evaluation.adjusted_rand_index.apply(data, kmeans_labels)
    assert 1 > kmeans_ari > 0


def test_ari_no_gt():
    # Create a dataset with no ground truth
    data = algpy.dataset.PointCloudDataset(np.asarray([[1, 2], [2, 3]]))

    # Try to evaluate with some labels
    labels = np.asarray([0, 1])
    with pytest.raises(ValueError, match="ground truth labels"):
        _ = algpy.evaluation.adjusted_rand_index.apply(data, labels)


def test_ari_wrong_dataset_type():
    # Create a non-clusterable dataset
    data = algpy.dataset.NoDataset()

    labels = np.asarray([0, 1])
    with pytest.raises(TypeError, match="dataset type"):
        _ = algpy.evaluation.adjusted_rand_index.apply(data, labels)


def test_ari_graph_dataset():
    # Create a graph dataset
    data = algpy.dataset.SBMDataset(100, 2, 0.5, 0.1)
    gt_labels = data.gt_labels
    ari = algpy.evaluation.adjusted_rand_index.apply(data, gt_labels)
    assert ari == 1


def test_num_vertices():
    num_vertices = algpy.evaluation.num_vertices.apply(algpy.dataset.NoDataset(),
                                                       stag.graph.complete_graph(100))
    assert num_vertices == 100

    # Num vertices works for any algpy dataset
    num_vertices = algpy.evaluation.num_vertices.apply(algpy.dataset.TwoMoonsDataset(),
                                                       stag.graph.complete_graph(100))
    assert num_vertices == 100


def test_avg_degree():
    avg_degree = algpy.evaluation.avg_degree.apply(algpy.dataset.NoDataset(),
                                                   stag.graph.cycle_graph(100))
    assert avg_degree == 2


def test_eigenvalue():
    graph = stag.random.sbm(100, 2, 0.8, 0.01)
    eig = algpy.evaluation.normalised_laplacian_second_eigenvalue.apply(algpy.dataset.NoDataset(),
                                                                        graph)
    assert 0.2 > eig > 0
