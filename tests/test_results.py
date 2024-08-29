"""Tests for the results module."""
from alglab.results import Results


def test_plots():
    # Run a simple experiment
    results = Results("results/results.csv")
    assert results.num_runs == 2

    results.line_plot('noise', 'running_time_s')

