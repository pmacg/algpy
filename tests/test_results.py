"""Tests for the results module."""
from alglab.results import Results
import alglab.algorithm
import alglab.experiment
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np


# We will use this KMeans implementation throughout the tests.
def kmeans(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_km = KMeans(n_clusters=k)
    sklearn_km.fit(data.data)
    return sklearn_km.labels_


def sc(data: alglab.dataset.PointCloudDataset, k=10):
    sklearn_sc = SpectralClustering(n_clusters=k)
    sklearn_sc.fit(data.data)
    return sklearn_sc.labels_


def test_plots():
    # Run a simple experiment
    results = Results("results/results.csv")
    assert results.num_runs == 2

    results.line_plot('noise', 'running_time_s')


def test_plots_multiple_parameters():
    alg1 = alglab.algorithm.Algorithm(kmeans)
    alg2 = alglab.algorithm.Algorithm(sc)

    noise_parameters = np.linspace(0, 1, 5)
    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        parameters={'k': 2,
                    'dataset.noise': noise_parameters,
                    'dataset.n': np.linspace(100, 1000, 3).astype(int)},
        evaluators=[alglab.evaluation.adjusted_rand_index]
    )
    results = experiments.run_all()
    results.line_plot('n', 'total_running_time_s', fixed_parameters={'noise': noise_parameters[0]})


