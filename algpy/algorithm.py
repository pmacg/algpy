"""
Create a generic class representing an algorithm which can be applied to a dataset.
"""

class  Algorithm(object):

    def __init__(self, alg):
        """Create the algorithm object. The alg parameter should be a function which
        takes a dataset and a list of parameters and runs the underlying algorithm we are studying."""
        self.inner_algorithm = alg

    def run(self, dataset, params):
        return self.inner_algorithm(dataset, **params)
