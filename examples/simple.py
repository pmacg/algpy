"""
A simple example demonstrating how to use AlgLab to compare two clustering algorithms.
"""
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import alglab


def main():
    # Configure the experiments. As well as the algorithms, we specify which dataset class to use,
    # and the parameters for the algorithms and dataset.
    #
    # The experiment schedule is given as the third argument and specifies which methods should be called on the
    # algorithm objects in order to run the experiment.
    #
    # We also specify any functions which should be used to evaluate the algorithms, and give a
    # filename in which to store the results.
    experiments = alglab.experiment.ExperimentalSuite(
        [KMeans, SpectralClustering],
        alglab.dataset.TwoMoonsDataset,
        alglab.experiment.StaticClusteringSchedule,
        "results/twomoonsresults.csv",
        parameters={
            "n_clusters": 2,
            "dataset.n": np.linspace(1000, 5000, 6).astype(int),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
    )

    # Run the experiments
    experiments.run_all()

    # Now, we can visualise the results
    results = alglab.results.Results("results/twomoonsresults.csv")
    results.line_plot("n", "total_running_time_s")
    results.line_plot("n", "adjusted_rand_index")


if __name__ == "__main__":
    main()
