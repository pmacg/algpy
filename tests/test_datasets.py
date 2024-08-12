import pytest
import numpy as np
import stag.graph
import algpy.dataset
import matplotlib.pyplot as plt


def test_no_dataset():
    # Check that initialising the dataset does not throw any error.
    _ = algpy.dataset.NoDataset()

    # Initialising with keyword arguments should give an error.
    with pytest.raises(Exception, match="keyword"):
        _ = algpy.dataset.NoDataset(eps=0.1)


def test_graph_dataset():
    graph = stag.graph.cycle_graph(100)
    dataset = algpy.dataset.GraphDataset(graph=graph)
    assert dataset.n == 100


def test_sbm_dataset():
    dataset = algpy.dataset.SBMDataset(n=100, k=2, p=0.5, q=0.1)
    assert dataset.n == 100

    with pytest.raises(Exception, match="p must be between 0 and 1"):
        _ = algpy.dataset.SBMDataset(n=100, k=2, p=-0.5, q=0.1)


def test_sbm_floats():
    # Check that we can initialise an SBM dataset with float arguments
    dataset = algpy.dataset.SBMDataset(n=100.0, k=2.1, p=0.5, q=0.1)
    assert dataset.n == 100


def test_pointcloud_dataset():
    data = np.asarray([[1, 1], [4, 2], [2, 3]])
    _ = algpy.dataset.PointCloudDataset(data=data)


def test_pointcloud_dataset_plot(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    data = np.asarray([[1, 1], [4, 2], [2, 3]])
    gt_labels = [0, 0, 1]
    dataset = algpy.dataset.PointCloudDataset(data=data, labels=gt_labels)
    dataset.plot_clusters(gt_labels)

    # Plotting a 3 dimensional dataset does not work
    with pytest.raises(Exception, match="two-dimensional"):
        data = np.asarray([[1, 1, 1], [2, 3, 3]])
        gt_labels = [0, 1]
        dataset = algpy.dataset.PointCloudDataset(data=data, labels=gt_labels)
        dataset.plot_clusters(gt_labels)

    # If the labels array is not the correct length, raises an exception
    with pytest.raises(Exception, match="length"):
        data = np.asarray([[1, 1], [4, 2], [2, 3]])
        dataset = algpy.dataset.PointCloudDataset(data=data)
        labels = [1, 1]
        dataset.plot_clusters(labels)


def test_twomoons_dataset(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)

    # Check that we can create the two moons dataset
    dataset = algpy.dataset.TwoMoonsDataset(n=200, noise=0.02)

    # Check that we can plot the dataset
    dataset.plot_clusters(dataset.gt_labels)