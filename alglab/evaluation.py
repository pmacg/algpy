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


def adjusted_rand_index(gt_labels, labels):
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

    if gt_labels is not None:
        return stag.cluster.adjusted_rand_index(gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')


def dataset_size(gt_labels, _):
    return len(gt_labels)


# -----------------------------------------------------------------------------
# Graph Evaluation
# -----------------------------------------------------------------------------

def num_vertices(_, graph: stag.graph.Graph):
    return graph.number_of_vertices()

def avg_degree(_, graph: stag.graph.Graph):
    return graph.average_degree()

def normalised_laplacian_second_eigenvalue(_, graph: stag.graph.Graph):
    lap = graph.normalised_laplacian().to_scipy()
    eigs, _ = scipy.sparse.linalg.eigsh(lap, which='SM', k=2)
    return eigs[1]
