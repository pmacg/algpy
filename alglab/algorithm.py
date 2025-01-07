"""
Create a generic class representing an algorithm which can be applied to a dataset.
"""
from typing import List, Type, Tuple, Callable, Dict, Union, get_type_hints
import inspect
import alglab.dataset
import time


class AlgorithmStep(object):
    def __init__(self,
                 name: str,
                 implementation: Callable,
                 first_step: bool):
        self.implementation = implementation
        self.name = name
        self.first_step = first_step

        # Automatically infer the return type
        self.return_type = self.__get_return_type()

        # Automatically infer the dataset class
        self.dataset_class = self.__get_dataset_class()

        # Automatically infer the parameter names for this algorithm step
        self.parameter_names = self.__get_parameter_names()

    def __get_return_type(self):
        sig = inspect.signature(self.implementation)
        if sig.return_annotation is not inspect._empty:
            return sig.return_annotation
        else:
            return object

    def __get_dataset_class(self):
        # Automatically infer the dataset class from type annotations
        sig = inspect.signature(self.implementation)

        non_default_parameters = [name for name, param in sig.parameters.items()
                                  if param.default == inspect.Parameter.empty]

        # Get the name of the dataset parameter
        if len(non_default_parameters) == 0:
            return alglab.dataset.NoDataset

        dataset_parameter = non_default_parameters[0]
        if not self.first_step:
            if len(non_default_parameters) == 0:
                raise ValueError("Second and later algorithm steps must take the output of the previous step as"
                                 " their first argument.")
            if len(non_default_parameters) == 1:
                return alglab.dataset.NoDataset
            if len(non_default_parameters) > 2:
                raise ValueError("All algorithm parameters must have default values.")
            dataset_parameter = non_default_parameters[1]
        else:
            if len(non_default_parameters) > 1:
                raise ValueError("All algorithm parameters must have default values.")

        # Extract the type hint for the dataset parameter, if one exists
        type_hints = get_type_hints(self.implementation)
        if dataset_parameter in type_hints:
            return type_hints[dataset_parameter]
        else:
            return alglab.dataset.Dataset

    def __get_parameter_names(self):
        inferred_parameter_names = []
        sig = inspect.signature(self.implementation)
        for name, param in sig.parameters.items():
            if param.default != inspect.Parameter.empty:
                if name not in inferred_parameter_names:
                    inferred_parameter_names.append(name)
        return inferred_parameter_names

    def run(self, dataset: alglab.dataset.Dataset, params: Dict, previous_step_output=None):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError("Provided dataset type must match dataset_class expected by the implementation.")

        # Filter the parameters by the ones accepted by this step
        this_step_parameters = {k: v for k, v in params.items() if k in self.parameter_names}

        non_kw_args = []
        if not self.first_step:
            non_kw_args.append(previous_step_output)
        if self.dataset_class is not alglab.dataset.NoDataset:
            non_kw_args.append(dataset)

        start_time = time.time()
        result = self.implementation(*non_kw_args, **this_step_parameters)
        end_time = time.time()
        running_time = end_time - start_time

        return result, running_time


class Algorithm(object):

    def __init__(self,
                 implementation: Union[Callable, List[Callable], List[Tuple[str, Callable]]],
                 name: str = None):
        """Create an algorithm definition. The implementation should be a python method which takes
        a dataset as a positional argument (if dataset_class is not NoDataset) and
        the parameters as keyword arguments. The implementation should return an object of type
        return_type.
        """
        self.implementation = []
        if isinstance(implementation, Callable):
            self.implementation = [AlgorithmStep(implementation.__name__, implementation, True)]
        elif isinstance(implementation, List):
            if len(implementation) == 0:
                raise ValueError("Algorithm must have at least one step.")
            if isinstance(implementation[0], Callable):
                first_step = True
                for imp in implementation:
                    self.implementation.append(AlgorithmStep(imp.__name__, imp, first_step))
                    first_step = False
            elif isinstance(implementation[0], Tuple):
                first_step = True
                for imp in implementation:
                    assert isinstance(imp, Tuple)
                    if len(imp) != 2:
                        raise ValueError("Algorithm steps should be tuples with the step name and implementation.")
                    self.implementation.append(AlgorithmStep(imp[0], imp[1], first_step))
                    first_step = False
        self.name = name if name is not None else self.implementation[0].name

        if self.name is "dataset":
            raise ValueError("It is not permitted to call an algorithm 'dataset'.")

        # Check for a return type hint
        self.return_type = self.implementation[-1].return_type

        # Automatically infer the dataset class if it is not provided
        self.dataset_class = self.implementation[0].dataset_class

        # Automatically infer the parameter names if they are not provided
        self.all_parameter_names = self.__get_parameters()

        # Check that the number of non-defaulted parameters in each step of the implementation is correct.
        self.__check_number_of_parameters()

        self.time_headings = [f'{step.name}_running_time_s' for step in self.implementation]
        self.time_headings.append('running_time_s')

    def __get_parameters(self):
        inferred_parameter_names = []
        for step in self.implementation:
            for par_name in step.parameter_names:
                if par_name not in inferred_parameter_names:
                    inferred_parameter_names.append(par_name)
        return inferred_parameter_names

    def __check_number_of_parameters(self):
        for step in self.implementation:
            if ((step.dataset_class is alglab.dataset.NoDataset and self.dataset_class is not alglab.dataset.NoDataset) or
                    (step.dataset_class is not alglab.dataset.NoDataset and self.dataset_class is alglab.dataset.NoDataset)):
                raise ValueError("All algorithm steps must take the dataset as an argument.")


    def run(self, dataset: alglab.dataset.Dataset, params: Dict):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError("Provided dataset type must match dataset_class expected by the implementation.")

        for param in params.keys():
            if param not in self.all_parameter_names:
                raise ValueError("Unexpected parameter name.")

        result = None
        running_times = {}
        start_time = time.time()
        for step in self.implementation:
            result, running_time = step.run(dataset, params, previous_step_output=result)
            running_times[f'{step.name}_running_time_s'] = running_time
        end_time = time.time()
        running_times['running_time_s'] = end_time - start_time

        if not isinstance(result, self.return_type):
            raise TypeError("Provided result type must match promised return_type.")

        return result, running_times

    def __repr__(self):
        return self.name
