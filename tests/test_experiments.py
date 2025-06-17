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


def dynamic_kmeans(data: alglab.dataset.DynamicPointCloudDataset, k=10):
    current_data = []
    for iteration in range(data.num_updates):
        insertions, _ = data.get_update(iteration)
        for new_index in insertions:
            current_data.append(data.data[new_index, :].tolist())
        sklearn_km = KMeans(n_clusters=k)
        sklearn_km.fit(current_data)
        yield sklearn_km.labels_


def dynamic_sc(data: alglab.dataset.DynamicPointCloudDataset, k=10):
    current_data = []
    for iteration in range(data.num_updates):
        insertions, _ = data.get_update(iteration)
        for new_index in insertions:
            current_data.append(data.data[new_index, :].tolist())
        sklearn_sc = SpectralClustering(n_clusters=k)
        sklearn_sc.fit(current_data)
        yield sklearn_sc.labels_


def test_experimental_suite():
    # Test the experimental suite class as it's intended to be used.
    alg1 = alglab.algorithm.Algorithm(kmeans)
    alg2 = alglab.algorithm.Algorithm(sc)

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
    alg1 = alglab.algorithm.Algorithm(kmeans)
    alg2 = alglab.algorithm.Algorithm(sc)

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


def test_memory_measurements():
    alg1 = alglab.algorithm.Algorithm(kmeans)
    alg2 = alglab.algorithm.Algorithm(sc)

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        parameters={
            "kmeans.k": 2,
            "sc.k": 2,
            "dataset.n": np.linspace(1000, 3000, 3),
            "dataset.noise": 0.5,
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
        num_runs=1,
    )
    results = experiments.run_all()
    results.line_plot('n', 'memory_usage_mib')


def test_dynamic_params():
    alg1 = alglab.algorithm.Algorithm(kmeans)
    alg2 = alglab.algorithm.Algorithm(sc)

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
    experiments.run_all()


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


def test_multi_step_algs():
    def kmeans_init(data: alglab.dataset.PointCloudDataset, k=10):
        return KMeans(n_clusters=k)

    def kmeans_predict(kmeans_obj, data: alglab.dataset.PointCloudDataset) -> np.ndarray:
        kmeans_obj.fit(data.data)
        return kmeans_obj.labels_

    def kmeans_slow_predict(kmeans_obj, data: alglab.dataset.PointCloudDataset) -> np.ndarray:
        kmeans_obj.fit(data.data)
        kmeans_obj.fit(data.data)
        return kmeans_obj.labels_

    alg1 = alglab.algorithm.Algorithm([('fit', kmeans_init), ('predict', kmeans_predict)], name='kmeans')
    alg2 = alglab.algorithm.Algorithm([('fit', kmeans_init), ('predict', kmeans_slow_predict)], name='slow_kmeans')

    experiments = alglab.experiment.ExperimentalSuite(
        [alg1, alg2],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        parameters={
            "k": 2,
            "dataset.n": np.linspace(100, 1000, 6),
        },
    )

    results = experiments.run_all()
    results.line_plot("n", "fit_running_time_s")
    results.line_plot("n", "predict_running_time_s")
    results.line_plot("n", "total_running_time_s")


def test_dynamic_algs():
    two_moons = alglab.dataset.TwoMoonsDataset()
    dynamic_two_moons = alglab.dataset.DynamicPointCloudDataset.from_pointcloud(two_moons, 100)

    assert(dynamic_two_moons.num_updates == 10)

    experiments = alglab.experiment.ExperimentalSuite([dynamic_kmeans, dynamic_sc],
                                                      dynamic_two_moons,
                                                      "results/dynamictwomoonsresults.csv",
                                                      evaluators=[alglab.evaluation.adjusted_rand_index,
                                                                  alglab.evaluation.dataset_size])
    results = experiments.run_all()

    # Plot the per-iteration running times for each algorithm
    results.line_plot("n", "total_running_time_s")
