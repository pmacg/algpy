"""Test the evaluation methods of alglab."""
import pytest
import stag.graph
import stag.random
from sklearn.cluster import KMeans
import numpy as np
import alglab.dataset
import alglab.evaluation


# We will use this KMeans implementation throughout the tests.
def kmeans_impl(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def test_ari():
    # Create a dataset with ground truth
    data = alglab.dataset.TwoMoonsDataset()

    best_ari = alglab.evaluation.adjusted_rand_index.apply(data, data.gt_labels)
    assert best_ari == 1

    kmeans_labels = kmeans_impl(data, k=2)
    kmeans_ari = alglab.evaluation.adjusted_rand_index.apply(data, kmeans_labels)
    assert 1 > kmeans_ari > 0


def test_ari_no_gt():
    # Create a dataset with no ground truth
    data = alglab.dataset.PointCloudDataset(np.asarray([[1, 2], [2, 3]]))

    # Try to evaluate with some labels
    labels = np.asarray([0, 1])
    with pytest.raises(ValueError, match="ground truth labels"):
        _ = alglab.evaluation.adjusted_rand_index.apply(data, labels)


def test_ari_wrong_dataset_type():
    # Create a non-clusterable dataset
    data = alglab.dataset.NoDataset()

    labels = np.asarray([0, 1])
    with pytest.raises(TypeError, match="dataset type"):
        _ = alglab.evaluation.adjusted_rand_index.apply(data, labels)


def test_ari_graph_dataset():
    # Create a graph dataset
    data = alglab.dataset.SBMDataset(100, 2, 0.5, 0.1)
    gt_labels = data.gt_labels
    ari = alglab.evaluation.adjusted_rand_index.apply(data, gt_labels)
    assert ari == 1


def test_num_vertices():
    num_vertices = alglab.evaluation.num_vertices.apply(alglab.dataset.NoDataset(),
                                                       stag.graph.complete_graph(100))
    assert num_vertices == 100

    # Num vertices works for any alglab dataset
    num_vertices = alglab.evaluation.num_vertices.apply(alglab.dataset.TwoMoonsDataset(),
                                                       stag.graph.complete_graph(100))
    assert num_vertices == 100


def test_avg_degree():
    avg_degree = alglab.evaluation.avg_degree.apply(alglab.dataset.NoDataset(),
                                                   stag.graph.cycle_graph(100))
    assert avg_degree == 2


def test_eigenvalue():
    graph = stag.random.sbm(100, 2, 0.8, 0.01)
    eig = alglab.evaluation.normalised_laplacian_second_eigenvalue.apply(alglab.dataset.NoDataset(),
                                                                        graph)
    assert 0.2 > eig > 0
