"""
Methods for evaluating the performance of an algorithm.
"""
from typing import Callable, Type
import numpy as np
import algpy.algorithm
import algpy.dataset
import stag.cluster
import stag.graph
import scipy.sparse.linalg


class Evaluator(object):

    def __init__(self,
                 implementation: Callable,
                 alg_result_type: Type = None,
                 dataset_class: Type[algpy.dataset.Dataset] = algpy.dataset.NoDataset):
        """Define a method of evaluating an algorithm. Specify the evaluator implementation as
        well as the expected result type of the algorithm to be evaluated and the type of the dataset
        the algorithm should be applied to."""
        self.implementation = implementation
        self.alg_result_type = alg_result_type
        self.dataset_class = dataset_class

    def apply(self, dataset: algpy.dataset.Dataset, alg_result):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError(f"Expected dataset type to be {self.dataset_class}, got {type(dataset)}.")

        if not isinstance(alg_result, self.alg_result_type):
            raise TypeError(f"Expected alg_result type to be {self.alg_result_type} but got {type(alg_result)}.")

        if self.dataset_class is not algpy.dataset.NoDataset:
            result = self.implementation(dataset, alg_result)
        else:
            result = self.implementation(alg_result)

        return result

# -----------------------------------------------------------------------------
# Clustering Evaluation
# -----------------------------------------------------------------------------


def ari_impl(data: algpy.dataset.ClusterableDataset, labels):
    if data.gt_labels is not None:
        return stag.cluster.adjusted_rand_index(data.gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')


adjusted_rand_index = Evaluator(ari_impl,
                                alg_result_type=np.ndarray,
                                dataset_class=algpy.dataset.ClusterableDataset)


# -----------------------------------------------------------------------------
# Graph Evaluation
# -----------------------------------------------------------------------------

def num_vertices_impl(_: algpy.dataset.Dataset, graph: stag.graph.Graph):
    return graph.number_of_vertices()


num_vertices = Evaluator(num_vertices_impl,
                         alg_result_type=stag.graph.Graph,
                         dataset_class=algpy.dataset.Dataset)


def avg_degree_impl(_: algpy.dataset.Dataset, graph: stag.graph.Graph):
    return graph.average_degree()


avg_degree = Evaluator(avg_degree_impl,
                       alg_result_type=stag.graph.Graph,
                       dataset_class=algpy.dataset.Dataset)


def normalised_laplacian_second_eigenvalue_impl(_: algpy.dataset.Dataset, graph: stag.graph.Graph):
    lap = graph.normalised_laplacian().to_scipy()
    eigs, _ = scipy.sparse.linalg.eigsh(lap, which='SM', k=2)
    return eigs[1]


normalised_laplacian_second_eigenvalue = Evaluator(normalised_laplacian_second_eigenvalue_impl,
                                                   alg_result_type=stag.graph.Graph,
                                                   dataset_class=algpy.dataset.Dataset)
