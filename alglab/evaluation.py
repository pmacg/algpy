"""
Methods for evaluating the performance of an algorithm.
"""
from typing import Callable, Type, get_type_hints, Union, List
import numpy as np
import alglab
import stag.cluster
import stag.graph
import scipy.sparse.linalg
import inspect


class Evaluator(object):

    def __init__(self,
                 implementation: Callable,
                 name: str = None,
                 alg_result_type: Type = None,
                 dataset_class: Type[alglab.dataset.Dataset] = None):
        """Define a method of evaluating an algorithm. Specify the evaluator implementation as
        well as the expected result type of the algorithm to be evaluated and the type of the dataset
        the algorithm should be applied to."""
        self.implementation = implementation
        sig = inspect.signature(self.implementation)

        self.name = name if name is not None else implementation.__name__

        # Automatically infer the dataset class from type annotations
        parameter_names = [
            name for name, param in sig.parameters.items()
        ]

        if len(parameter_names) < 1:
            raise ValueError("Evaluator implementation must take at least one parameter.")
        if len(parameter_names) > 2:
            raise ValueError("Evaluator implementation must take at most two parameters: (optionally) a dataset, "
                             "and the algorithm's output.")

        self.dataset_class = dataset_class
        if self.dataset_class is None:
            # If there is only one positional argument to the implementation, then there is no dataset.
            if len(parameter_names) == 1:
                self.dataset_class = alglab.dataset.NoDataset
            else:
                # Get the name of the dataset parameter
                dataset_parameter = parameter_names[0]

                # Extract the type hint for the dataset parameter, if one exists
                type_hints = get_type_hints(implementation)
                if dataset_parameter in type_hints:
                    self.dataset_class = type_hints[dataset_parameter]
                else:
                    self.dataset_class = alglab.dataset.Dataset

        self.alg_result_type = alg_result_type
        if self.alg_result_type is None:
            # Get the name of the result parameter
            result_parameter = parameter_names[-1]

            # Extract the type hint for the result parameter, if it exists
            type_hints = get_type_hints(implementation)
            if result_parameter in type_hints:
                self.alg_result_type = type_hints[result_parameter]
            else:
                self.alg_result_type = object

    def apply(self, dataset: alglab.dataset.Dataset, alg_result):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError(f"Alglab Evaluation Error: Expected dataset type to be {self.dataset_class}, got {type(dataset)}.")

        if not isinstance(alg_result, self.alg_result_type):
            raise TypeError(f"Alglab Evaluation Error: expected algorithm output type to be {self.alg_result_type} but got {type(alg_result)}.")

        if self.dataset_class is not alglab.dataset.NoDataset:
            result = self.implementation(dataset, alg_result)
        else:
            result = self.implementation(alg_result)

        return result

    def __str__(self):
        return self.name

# -----------------------------------------------------------------------------
# Clustering Evaluation
# -----------------------------------------------------------------------------


def __ari_impl(data: alglab.dataset.ClusterableDataset, labels):
    # Remove any negative cluster ids
    next_cluster_id = np.max(labels) + 1
    mapping = {}
    indices_to_change = []
    for i, label in enumerate(labels):
        if label < 0:
            if label not in mapping:
                mapping[label] = next_cluster_id
                next_cluster_id += 1
            indices_to_change.append(i)
    for i in indices_to_change:
        labels[i] = mapping[labels[i]]

    if data.gt_labels is not None:
        return stag.cluster.adjusted_rand_index(data.gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')


adjusted_rand_index = Evaluator(__ari_impl,
                                name="adjusted_rand_index",
                                alg_result_type=np.ndarray,
                                dataset_class=alglab.dataset.ClusterableDataset)


# -----------------------------------------------------------------------------
# Helpful hacks
# -----------------------------------------------------------------------------
def __n_impl(data: alglab.dataset.PointCloudDataset, _):
    return data.n


dataset_size = Evaluator(__n_impl,
                         name="n",
                         alg_result_type=object,
                         dataset_class=alglab.dataset.PointCloudDataset)

# -----------------------------------------------------------------------------
# Graph Evaluation
# -----------------------------------------------------------------------------

def __num_vertices_impl(_: alglab.dataset.Dataset, graph: stag.graph.Graph):
    return graph.number_of_vertices()


num_vertices = Evaluator(__num_vertices_impl,
                         name="number_of_vertices",
                         alg_result_type=stag.graph.Graph,
                         dataset_class=alglab.dataset.Dataset)


def __avg_degree_impl(_: alglab.dataset.Dataset, graph: stag.graph.Graph):
    return graph.average_degree()


avg_degree = Evaluator(__avg_degree_impl,
                       name="average_degree",
                       alg_result_type=stag.graph.Graph,
                       dataset_class=alglab.dataset.Dataset)


def __normalised_laplacian_second_eigenvalue_impl(_: alglab.dataset.Dataset, graph: stag.graph.Graph):
    lap = graph.normalised_laplacian().to_scipy()
    eigs, _ = scipy.sparse.linalg.eigsh(lap, which='SM', k=2)
    return eigs[1]


normalised_laplacian_second_eigenvalue = Evaluator(__normalised_laplacian_second_eigenvalue_impl,
                                                   name="lap_second_eigenvalue",
                                                   alg_result_type=stag.graph.Graph,
                                                   dataset_class=alglab.dataset.Dataset)
