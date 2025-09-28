"""Tests for the results module."""
from alglab.results import Results
import alglab.experiment
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import sys


def test_plots():
    # Run a simple experiment
    results = Results("results/results.csv")
    assert results.num_runs == 2
    results.line_plot('noise', 'total_running_time_s',
                      fixed_parameters={'n': 1000})


def test_plots_multiple_parameters():
    noise_parameters = np.linspace(0, 1, 5)
    experiments = alglab.experiment.ExperimentalSuite(
        [KMeans, SpectralClustering],
        alglab.dataset.TwoMoonsDataset,
        alglab.experiment.StaticClusteringSchedule,
        "results/twomoonsresults.csv",
        parameters={'n_clusters': 2,
                    'dataset.noise': noise_parameters,
                    'dataset.n': np.linspace(100, 1000, 3).astype(int)},
        evaluators=[alglab.evaluation.adjusted_rand_index],
        num_runs=2,
    )
    results = experiments.run_all()
    results.line_plot('n', 'total_running_time_s', fixed_parameters={'noise': noise_parameters[0]})


