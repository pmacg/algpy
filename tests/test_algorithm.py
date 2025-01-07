"""Tests for the algorithm class."""
import pytest

import alglab.dataset
import alglab.algorithm
import alglab.evaluation
from sklearn.cluster import KMeans
import numpy as np
from typing import List


def kmeans(data: alglab.dataset.PointCloudDataset, k=10) -> np.ndarray:
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def kmeans_no_return(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def no_dataset_impl(n=10):
    return n ** 3


def test_algorithm():
    # Test the algorithm class as it is intended to be used.
    kmeans_algorithm = alglab.algorithm.Algorithm(kmeans,
                                                  name='kmeans')

    test_data = alglab.dataset.TwoMoonsDataset()
    _ = kmeans_algorithm.run(test_data, {'k': 8})


def test_fit_predict_algorithm():
    def kmeans_init(data: alglab.dataset.PointCloudDataset, k=10):
        return KMeans(n_clusters=k)

    def kmeans_predict(kmeans_obj, data: alglab.dataset.PointCloudDataset) -> np.ndarray:
        kmeans_obj.fit(data.data)
        return kmeans_obj.labels_

    kmeans_algorithm = alglab.algorithm.Algorithm([kmeans_init, kmeans_predict],
                                                  name='kmeans')

    test_data = alglab.dataset.TwoMoonsDataset()
    labels, times = kmeans_algorithm.run(test_data, {'k': 8})
    assert(len(times.keys()) == 3)


def test_no_return_type():
    kmeans_algorithm = alglab.algorithm.Algorithm(kmeans_no_return)
    test_data = alglab.dataset.TwoMoonsDataset()
    _ = kmeans_algorithm.run(test_data, {'k': 8})


def test_bad_specified_return_type():
    # Create an implementation with the wrong return type hint
    def bad_return(n=10) -> np.ndarray:
        return n ** 3

    alg = alglab.algorithm.Algorithm(bad_return)

    with pytest.raises(TypeError, match='return'):
        _ = alg.run(alglab.dataset.NoDataset(), {'n': 20})


def test_infer_parameters_with_dataset():
    kmeans_algorithm = alglab.algorithm.Algorithm(kmeans)

    test_data = alglab.dataset.TwoMoonsDataset()
    _ = kmeans_algorithm.run(test_data, {'k': 8})

    with pytest.raises(ValueError, match='parameter'):
        _ = kmeans_algorithm.run(test_data, {'data': test_data})


def test_all_params_default():
    def non_defaults(k, l, n=10) -> int:
        return k * l * n ** 3

    with pytest.raises(ValueError, match='default'):
        alg = alglab.algorithm.Algorithm(non_defaults)


def test_pass_wrong_dataset_type():
    kmeans_algorithm = alglab.algorithm.Algorithm(kmeans)

    test_data = alglab.dataset.GraphDataset()
    with pytest.raises(TypeError, match='dataset'):
        _ = kmeans_algorithm.run(test_data, {'k': 8})


def test_no_dataset_type_hint():
    def kmeans_nohint(data, k=10):
        sklearn_km = KMeans(n_clusters=k)
        sklearn_km.fit(data.data)
        return sklearn_km.labels_

    alg = alglab.algorithm.Algorithm(kmeans_nohint)
    test_data = alglab.dataset.TwoMoonsDataset()
    _ = alg.run(test_data, {'k': 8})


def test_all_type_hints():
    def kmeans_hints(data: alglab.dataset.PointCloudDataset, k: int = 10):
        sklearn_km = KMeans(n_clusters=k)
        sklearn_km.fit(data.data)
        return sklearn_km.labels_

    alg = alglab.algorithm.Algorithm(kmeans_hints)
    test_data = alglab.dataset.TwoMoonsDataset()
    _ = alg.run(test_data, {'k': 8})


def test_automatically_infer_parameters_bad():
    def good_return(n=10) -> int:
        return n ** 3

    alg = alglab.algorithm.Algorithm(good_return)

    with pytest.raises(ValueError, match='parameter'):
        _ = alg.run(alglab.dataset.NoDataset(), {'b': 20})


def test_unspecified_parameter():
    # Test the algorithm class as it is intended to be used.
    kmeans_algorithm = alglab.algorithm.Algorithm(kmeans)

    # Not fully specifying the parameter is permitted when running.
    test_data = alglab.dataset.TwoMoonsDataset()
    _ = kmeans_algorithm.run(test_data, {})


def test_no_dataset():
    cube_algorithm = alglab.algorithm.Algorithm(no_dataset_impl,
                                                name="power")

    _ = cube_algorithm.run(alglab.dataset.NoDataset(), {'n': 20})


def test_wrong_dataset():
    cube_algorithm = alglab.algorithm.Algorithm(no_dataset_impl,
                                                name="power")

    with pytest.raises(TypeError, match='dataset'):
        _ = cube_algorithm.run(alglab.dataset.TwoMoonsDataset(), {'n': 20})


def test_openml_dataset():
    dataset = alglab.dataset.OpenMLDataset(name="iris")
    kmeans_algorithm = alglab.algorithm.Algorithm(kmeans)

    labels, _ = kmeans_algorithm.run(dataset, {'k': 3})

    # Check that evaluation works
    _ = alglab.evaluation.adjusted_rand_index.apply(dataset, labels)
