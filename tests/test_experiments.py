"""Tests for the experiment module."""
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import algpy.dataset
import algpy.experiment
import algpy.evaluation
import algpy.algorithm


# We will use this KMeans implementation throughout the tests.
def kmeans_impl(data: algpy.dataset.PointCloudDataset, k=10):
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def sc_impl(data: algpy.dataset.PointCloudDataset, k=10):
    sklearn_sc = SpectralClustering(n_clusters=k)
    sklearn_sc.fit(data.data)
    return sklearn_sc.labels_


def test_experimental_suite():
    # Test the experimental suite class as it's intended to be used.
    alg1 = algpy.algorithm.Algorithm("kmeans",
                                     kmeans_impl,
                                     np.ndarray,
                                     ["k"],
                                     algpy.dataset.PointCloudDataset)
    alg2 = algpy.algorithm.Algorithm("sc",
                                     sc_impl,
                                     np.ndarray,
                                     ["k"],
                                     algpy.dataset.PointCloudDataset)

    experiments = algpy.experiment.ExperimentalSuite(
        [alg1, alg2],
        algpy.dataset.TwoMoonsDataset,
        "twomoonsresults.csv",
        alg_fixed_params={'kmeans': {'k': 2}, 'sc': {'k': 2}},
        dataset_fixed_params={'n': 1000},
        dataset_varying_params={'noise': np.linspace(0, 1, 5)},
        evaluators=[algpy.evaluation.adjusted_rand_index]
        )
    experiments.run_all()