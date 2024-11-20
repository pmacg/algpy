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

    algs = [alglab.algorithm.Algorithm(kmeans),
            alglab.algorithm.Algorithm(spectral_clustering)]

    experiments = alglab.experiment.ExperimentalSuite(
        algorithms=algs,
        dataset=alglab.dataset.TwoMoonsDataset,
        parameters={
            "k": 2,
            "dataset.n": 1000,
            "dataset.noise": np.linspace(0, 1, 5),
        },
        evaluators=[alglab.evaluation.adjusted_rand_index],
        results_filename="results/twomoonsresults.csv"
    )
    experiments.run_all()


if __name__ == "__main__":
    main()
