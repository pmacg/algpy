"""
Methods for evaluating the performance of an algorithm.
"""
import algpy.algorithm
import algpy.dataset
import stag.cluster

def adjusted_rand_index(data: algpy.dataset.PointCloudDataset, labels):
    if data.gt_labels is not None:
        return stag.cluster.adjusted_rand_index(data.gt_labels, labels)
    else:
        raise ValueError('No ground truth labels provided.')
