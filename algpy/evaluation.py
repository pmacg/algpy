"""
Methods for evaluating the performance of an algorithm.
"""
import algpy.algorithm
import algpy.dataset
import stag.cluster
import stag.graph
import scipy.sparse.linalg


def adjusted_rand_index(data: algpy.dataset.PointCloudDataset, labels):
    if data.gt_labels is not None:
        return stag.cluster.adjusted_rand_index(data.gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')


def num_vertices(_: algpy.dataset.Dataset, graph: stag.graph.Graph):
    return graph.number_of_vertices()


def avg_degree(_: algpy.dataset.Dataset, graph: stag.graph.Graph):
    return graph.average_degree()


def normalised_laplacian_second_eigenvalue(_: algpy.dataset.Dataset, graph: stag.graph.Graph):
    lap = graph.normalised_laplacian().to_scipy()
    eigs, _ = scipy.sparse.linalg.eigsh(lap, which='SM', k=2)
    return eigs[1]
