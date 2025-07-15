"""Tests for the experiment module."""
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import pytest
import alglab

class DynamicKMeans(object):

    def __init__(self, k=10):
        self.current_data = []
        self.k = k

    def add_point(self, new_point: np.ndarray):
        self.current_data.append(new_point.tolist())

    def predict(self):
        sklearn_km = KMeans(n_clusters=self.k)
        sklearn_km.fit(self.current_data)
        return sklearn_km.labels_


class DynamicSC(object):

    def __init__(self, k=10):
        self.current_data = []
        self.k = k

    def add_point(self, new_point: np.ndarray):
        self.current_data.append(new_point.tolist())

    def predict(self):
        sklearn_km = SpectralClustering(n_clusters=self.k)
        sklearn_km.fit(self.current_data)
        return sklearn_km.labels_


def test_experimental_suite():
    experiments = alglab.experiment.ExperimentalSuite(
        [KMeans, SpectralClustering],
        alglab.dataset.TwoMoonsDataset,
        alglab.experiment.StaticClusteringSchedule,
        "results/twomoonsresults.csv",
        parameters={
            "KMeans.n_clusters": 2,
            "SpectralClustering.n_clusters": 2,
            "dataset.n": 1000,
            "dataset.noise": np.linspace(0, 1, 5),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index]
        )
    experiments.run_all()


def test_multiple_runs():
    # Test the experimental suite class as it's intended to be used.
    experiments = alglab.experiment.ExperimentalSuite(
        [KMeans, SpectralClustering],
        alglab.dataset.TwoMoonsDataset,
        alglab.experiment.StaticClusteringSchedule,
        "results/twomoonsresults.csv",
        parameters={
            "KMeans.n_clusters": 2,
            "SpectralClustering.n_clusters": 2,
            "dataset.n": 1000,
            "dataset.noise": np.linspace(0, 1, 5),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
        num_runs=2
    )
    assert experiments.num_trials == 20
    experiments.run_all()


def test_memory_measurements():
    experiments = alglab.experiment.ExperimentalSuite(
        [KMeans, SpectralClustering],
        alglab.dataset.TwoMoonsDataset,
        alglab.experiment.StaticClusteringSchedule,
        "results/twomoonsresults.csv",
        parameters={
            "KMeans.n_clusters": 2,
            "SpectralClustering.n_clusters": 2,
            "dataset.n": np.linspace(1000, 3000, 3),
            "dataset.noise": 0.5,
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
        num_runs=1,
    )
    results = experiments.run_all()
    results.line_plot('n', 'memory_usage_mib')


def test_dynamic_params():
    experiments = alglab.experiment.ExperimentalSuite(
        [KMeans, SpectralClustering],
        alglab.dataset.TwoMoonsDataset,
        alglab.experiment.StaticClusteringSchedule,
        "results/twomoonsresults.csv",
        parameters={
            "KMeans.n_clusters": 2,
            "SpectralClustering.n_clusters": [(lambda p: int(p['n'] / 100)), 2],
            "dataset.noise": 0.1,
            "dataset.n": np.linspace(100, 1000, 5).astype(int),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index]
    )
    experiments.run_all()


def test_simple_configuration():
    experiments = alglab.experiment.ExperimentalSuite([KMeans, SpectralClustering],
                                                      alglab.dataset.TwoMoonsDataset,
                                                      alglab.experiment.StaticClusteringSchedule,
                                                      "results/twomoonsresults.csv")
    experiments.run_all()


def test_simple_with_custom_evaluator():
    def const_evaluator(expected_output, alg_output):
        return 5

    experiments = alglab.experiment.ExperimentalSuite([KMeans, SpectralClustering],
                                                      alglab.dataset.TwoMoonsDataset,
                                                      alglab.experiment.StaticClusteringSchedule,
                                                      "results/twomoonsresults.csv",
                                                      evaluators=[const_evaluator])
    experiments.run_all()


def test_wrong_alg_name():
    algs = [KMeans, SpectralClustering]

    with pytest.raises(ValueError, match='algorithm'):
        experiments = alglab.experiment.ExperimentalSuite(
            algs,
            alglab.dataset.TwoMoonsDataset,
            alglab.experiment.StaticClusteringSchedule,
            "results/twomoonsresults.csv",
            parameters={
                "spectral_clustering.k": 2,
                "dataset.n": 1000,
                "dataset.noise": np.linspace(0, 1, 5),
            },
            evaluators=[alglab.evaluation.adjusted_rand_index]
        )


def test_multi_step_algs():
    experiments = alglab.experiment.ExperimentalSuite(
        [KMeans, GaussianMixture],
        alglab.dataset.TwoMoonsDataset,
        alglab.experiment.FitPredictSchedule,
        "results/twomoonsresults.csv",
        parameters={
            "KMeans.n_clusters": 2,
            "GaussianMixture.n_components": 2,
            "dataset.n": np.linspace(100, 1000, 6),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
    )

    results = experiments.run_all()
    results.line_plot("n", "fit_total_running_time_s")
    results.line_plot("n", "predict_total_running_time_s")
    results.line_plot("n", "total_running_time_s")


def test_dynamic_algs():
    experiments = alglab.experiment.ExperimentalSuite([DynamicKMeans, DynamicSC],
                                                      alglab.dataset.TwoMoonsDataset,
                                                      alglab.experiment.DynamicClusteringSchedule,
                                                      "results/dynamictwomoonsresults.csv",
                                                      parameters={
                                                          "DynamicKMeans.k": 2,
                                                          "DynamicSC.k": 2,
                                                          "schedule.batch_size": 100,
                                                      },
                                                      evaluators=[alglab.evaluation.adjusted_rand_index,
                                                                  alglab.evaluation.dataset_size])
    results = experiments.run_all()

    # Plot the per-iteration running times for each algorithm
    results.line_plot("dataset_size", "total_running_time_s")

