"""Tests for the experiment module."""
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import pytest
import alglab


# We will use this KMeans implementation throughout the tests.
def kmeans(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def sc(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_sc = SpectralClustering(n_clusters=k)
    sklearn_sc.fit(data.data)
    return sklearn_sc.labels_


def test_experimental_suite():
    # Test the experimental suite class as it's intended to be used.
    alg1 = alglab.algorithm.Algorithm(kmeans,
                                      return_type=np.ndarray,
                                      parameter_names=["k"],
                                      dataset_class=alglab.dataset.PointCloudDataset)
    alg2 = alglab.algorithm.Algorithm(sc,
                                      return_type=np.ndarray,
                                      parameter_names=["k"],
                                      dataset_class=alglab.dataset.PointCloudDataset)

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        parameters={
            "kmeans.k": 2,
            "sc.k": 2,
            "dataset.n": 1000,
            "dataset.noise": np.linspace(0, 1, 5),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index]
        )
    experiments.run_all()


def test_multiple_runs():
    # Test the experimental suite class as it's intended to be used.
    alg1 = alglab.algorithm.Algorithm(kmeans,
                                      return_type=np.ndarray,
                                      parameter_names=["k"],
                                      dataset_class=alglab.dataset.PointCloudDataset)
    alg2 = alglab.algorithm.Algorithm(sc,
                                      return_type=np.ndarray,
                                      parameter_names=["k"],
                                      dataset_class=alglab.dataset.PointCloudDataset)

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        parameters={
            "kmeans.k": 2,
            "sc.k": 2,
            "dataset.n": 1000,
            "dataset.noise": np.linspace(0, 1, 5),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
        num_runs=2
    )
    assert experiments.num_trials == 20
    experiments.run_all()


def test_dynamic_params():
    alg1 = alglab.algorithm.Algorithm(kmeans,
                                      return_type=np.ndarray,
                                      parameter_names=["k"],
                                      dataset_class=alglab.dataset.PointCloudDataset)
    alg2 = alglab.algorithm.Algorithm(sc,
                                      return_type=np.ndarray,
                                      parameter_names=["k"],
                                      dataset_class=alglab.dataset.PointCloudDataset)

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        parameters={
            "kmeans.k": 2,
            "sc.k": [(lambda p: int(p['n'] / 100)), 2],
            "dataset.noise": 0.1,
            "dataset.n": np.linspace(100, 1000, 5).astype(int),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index]
    )
    experiments.run_all()


def test_simple_configuration():
    experiments = alglab.experiment.ExperimentalSuite([kmeans, sc],
                                                      alglab.dataset.TwoMoonsDataset,
                                                      "results/twomoonsresults.csv")
    experiments.run_all()


def test_simple_with_custom_evaluator():
    def const_evaluator(data, alg_output):
        return 5

    experiments = alglab.experiment.ExperimentalSuite([kmeans, sc],
                                                      alglab.dataset.TwoMoonsDataset,
                                                      "results/twomoonsresults.csv",
                                                      evaluators=[const_evaluator])


def test_wrong_alg_name():
    algs = [alglab.algorithm.Algorithm(kmeans),
            alglab.algorithm.Algorithm(sc)]

    with pytest.raises(ValueError, match='algorithm'):
        experiments = alglab.experiment.ExperimentalSuite(
            algs,
            alglab.dataset.TwoMoonsDataset,
            "results/twomoonsresults.csv",
            parameters={
                "spectral_clustering.k": 2,
                "dataset.n": 1000,
                "dataset.noise": np.linspace(0, 1, 5),
            },
            evaluators=[alglab.evaluation.adjusted_rand_index]
        )
