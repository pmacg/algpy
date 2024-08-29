"""Tests for the experiment module."""
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import alglab.dataset
import alglab.experiment
import alglab.evaluation
import alglab.algorithm


# We will use this KMeans implementation throughout the tests.
def kmeans_impl(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def sc_impl(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_sc = SpectralClustering(n_clusters=k)
    sklearn_sc.fit(data.data)
    return sklearn_sc.labels_


def test_experimental_suite():
    # Test the experimental suite class as it's intended to be used.
    alg1 = alglab.algorithm.Algorithm("kmeans",
                                     kmeans_impl,
                                     np.ndarray,
                                     ["k"],
                                     alglab.dataset.PointCloudDataset)
    alg2 = alglab.algorithm.Algorithm("sc",
                                     sc_impl,
                                     np.ndarray,
                                     ["k"],
                                     alglab.dataset.PointCloudDataset)

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        alg_fixed_params={'kmeans': {'k': 2}, 'sc': {'k': 2}},
        dataset_fixed_params={'n': 1000},
        dataset_varying_params={'noise': np.linspace(0, 1, 5)},
        evaluators=[alglab.evaluation.adjusted_rand_index]
        )
    experiments.run_all()


def test_multiple_runs():
    # Test the experimental suite class as it's intended to be used.
    alg1 = alglab.algorithm.Algorithm("kmeans",
                                     kmeans_impl,
                                     np.ndarray,
                                     ["k"],
                                     alglab.dataset.PointCloudDataset)
    alg2 = alglab.algorithm.Algorithm("sc",
                                     sc_impl,
                                     np.ndarray,
                                     ["k"],
                                     alglab.dataset.PointCloudDataset)

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        alg_fixed_params={'kmeans': {'k': 2}, 'sc': {'k': 2}},
        dataset_fixed_params={'n': 1000},
        dataset_varying_params={'noise': np.linspace(0, 1, 5)},
        evaluators=[alglab.evaluation.adjusted_rand_index],
        num_runs=2
    )
    assert experiments.num_trials == 20
    experiments.run_all()


def test_dynamic_params():
    alg1 = alglab.algorithm.Algorithm("kmeans",
                                     kmeans_impl,
                                     np.ndarray,
                                     ["k"],
                                     alglab.dataset.PointCloudDataset)
    alg2 = alglab.algorithm.Algorithm("sc",
                                     sc_impl,
                                     np.ndarray,
                                     ["k"],
                                     alglab.dataset.PointCloudDataset)

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        alg_fixed_params={'kmeans': {'k': 2}},
        alg_varying_params={'sc': {'k': [(lambda p: int(p['n'] / 100)), 2]}},
        dataset_fixed_params={'noise': 0.1},
        dataset_varying_params={'n': np.linspace(100, 1000, 5).astype(int)},
        evaluators=[alglab.evaluation.adjusted_rand_index]
    )
    experiments.run_all()

