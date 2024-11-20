"""
Create a generic class representing an algorithm which can be applied to a dataset.
"""
from typing import List, Type, Callable, Dict, Any
import inspect
import alglab.dataset


class Algorithm(object):

    def __init__(self,
                 implementation: Callable,
                 name: str = None,
                 return_type: Type = object,
                 parameter_names: List[str] = None,
                 dataset_class: Type[alglab.dataset.Dataset] = alglab.dataset.NoDataset):
        """Create an algorithm definition. The implementation should be a python method which takes
        a dataset as a positional argument (if dataset_class is not NoDataset) and
        the parameters as keyword arguments. The implementation should return an object of type
        return_type.
        """
        self.implementation = implementation

        sig = inspect.signature(self.implementation)

        # Check for a return type hint
        self.return_type = return_type
        if self.return_type is object and sig.return_annotation is not inspect._empty:
            self.return_type = sig.return_annotation

        # Automatically infer the parameter names if they are not provided
        self.parameter_names = []
        if parameter_names is None:
            self.parameter_names = [
                name for name, param in sig.parameters.items()
                if param.default != inspect.Parameter.empty
            ]
        else:
            self.parameter_names = parameter_names

        # Check that the parameters of the algorithm have default values
        if len(self.parameter_names) < len(sig.parameters.items()) - 1:
            raise ValueError("All algorithm parameters should have a default value.")

        self.dataset_class = dataset_class
        self.name = name if name is not None else implementation.__name__

    def run(self, dataset: alglab.dataset.Dataset, params: Dict):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError("Provided dataset type must match dataset_class expected by the implementation.")

        for param in params.keys():
            if param not in self.parameter_names:
                raise ValueError("Unexpected parameter name.")

        if self.dataset_class is not alglab.dataset.NoDataset:
            result = self.implementation(dataset, **params)
        else:
            result = self.implementation(**params)

        if not isinstance(result, self.return_type):
            raise TypeError("Provided result type must match promised return_type.")

        return result

    def __repr__(self):
        return self.name
