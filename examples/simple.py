"""
A simple example demonstrating how to use AlgLab to compare two clustering algorithms.
"""
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import alglab


def main():
    # First, implement the algorithms that you would like to compare.
    # Note that the signature of the implemented algorithms should take a dataset as the first argument,
    # followed by the algorithm parameters as keyword arguments, with default values.
    def kmeans(data: alglab.dataset.PointCloudDataset, k=10):
        sklearn_km = KMeans(n_clusters=k)
        sklearn_km.fit(data.data)
        return sklearn_km.labels_

    def spectral_clustering(data: alglab.dataset.PointCloudDataset, k=10):
        sklearn_sc = SpectralClustering(n_clusters=k)
        sklearn_sc.fit(data.data)
        return sklearn_sc.labels_

    # Configure the experiments. As well as the algorithms, we specify which dataset class to use,
    # and the parameters for the algorithms and dataset.
    #
    # We also specify any functions which should be used to evaluate the algorithms, and give a
    # filename in which to store the results.
    experiments = alglab.experiment.ExperimentalSuite(
        [kmeans, spectral_clustering],
        alglab.dataset.TwoMoonsDataset,
        "results/twomoonsresults.csv",
        parameters={
            "k": 2,
            "dataset.n": np.linspace(1000, 5000, 6).astype(int),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
    )

    # Run the experiments
    experiments.run_all()

    # Now, we can visualise the results
    results = alglab.results.Results("results/twomoonsresults.csv")
    results.line_plot("n", "running_time_s")
    results.line_plot("n", "adjusted_rand_index")


if __name__ == "__main__":
    main()
