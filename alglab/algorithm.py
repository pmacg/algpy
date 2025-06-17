"""
Create a generic class representing an algorithm which can be applied to a dataset.
"""
import threading
from typing import List, Type, Tuple, Callable, Dict, Union, get_type_hints
from types import GeneratorType
import inspect
import alglab.dataset
import time
import psutil
import os

def monitor_memory(interval, stop_event, result_dict, base_memory):
    process = psutil.Process(os.getpid())
    peak_diff = 0
    while not stop_event.is_set():
        current = process.memory_info().rss
        diff = current - base_memory
        peak_diff = max(peak_diff, diff)
        time.sleep(interval)
    result_dict['peak_diff'] = peak_diff


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

        return self.implementation(*non_kw_args, **this_step_parameters)


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

        self.number_of_steps = len(self.implementation)

        # Check for a return type hint
        self.return_type = self.implementation[-1].return_type

        # Automatically infer the dataset class if it is not provided
        self.dataset_class = self.implementation[0].dataset_class

        # Automatically infer the parameter names if they are not provided
        self.all_parameter_names = self.__get_parameters()

        # Check that the number of non-defaulted parameters in each step of the implementation is correct.
        self.__check_number_of_parameters()

        self.results_headings = []
        if self.number_of_steps > 1:
            for step in self.implementation:
                self.results_headings.append(f'{step.name}_running_time_s')
        else:
            self.results_headings.append('iter')
            self.results_headings.append('iter_running_time_s')
        self.results_headings.append('total_running_time_s')
        self.results_headings.append('memory_usage_mib')

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

    def __run_multi_step(self, dataset: alglab.dataset.Dataset, params: Dict):
        result = None
        running_times = {}
        global_start_time = time.time()
        for step in self.implementation:
            start_time = time.time()
            result = step.run(dataset, params, previous_step_output=result)
            end_time = time.time()
            running_times[f'{step.name}_running_time_s'] = end_time - start_time
        end_time = time.time()
        running_times['total_running_time_s'] = end_time - global_start_time

        return result, running_times

    def __run_single_step(self, dataset: alglab.dataset.Dataset, params: Dict):
        # Set up the memory tracking
        this_process = psutil.Process(os.getpid())
        base_memory = this_process.memory_info().rss
        memory_dict = {}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_memory,
            args=(0.1, stop_event, memory_dict, base_memory)
        )

        # Run the algorithm and measure time and memory usage
        monitor_thread.start()
        start_time = time.time()
        experiment_gen = self.implementation[0].run(dataset, params)
        end_time = time.time()
        stop_event.set()
        monitor_thread.join()
        peak_memory_bytes = memory_dict.get('peak_diff', 0)

        if not isinstance(experiment_gen, GeneratorType):
            # This is not a dynamic algorithm - return the result
            total_running_time = end_time - start_time
            result = {'iter': 0,
                      'iter_running_time_s': end_time - start_time,
                      'total_running_time_s': total_running_time,
                      'memory_usage_mib': peak_memory_bytes / 1024 / 1024,}
            yield experiment_gen, result
        else:
            run_ended = False
            total_running_time = 0
            iteration = 0
            while not run_ended:
                monitor_thread = threading.Thread(
                    target=monitor_memory,
                    args=(0.1, stop_event, memory_dict, base_memory)
                )
                monitor_thread.start()
                try:
                    start_time = time.time()
                    alg_output = next(experiment_gen)
                    end_time = time.time()
                    stop_event.set()
                    monitor_thread.join()
                    peak_memory_bytes = memory_dict.get('peak_diff', 0)

                    total_running_time += end_time - start_time
                    result = {'iter': iteration,
                              'iter_running_time_s': end_time - start_time,
                              'total_running_time_s': total_running_time,
                              'memory_usage_mib': peak_memory_bytes / 1024 / 1024,}
                    yield alg_output, result
                    iteration += 1
                except StopIteration:
                    stop_event.set()
                    monitor_thread.join()
                    run_ended = True

    def run_static(self, dataset: alglab.dataset.Dataset, params: Dict):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError("Provided dataset type must match dataset_class expected by the implementation.")

        for param in params.keys():
            if param not in self.all_parameter_names:
                raise ValueError("Unexpected parameter name.")

        last_output = None
        last_results = None
        if self.number_of_steps > 1:
            last_output, last_results = self.__run_multi_step(dataset, params)
        else:
            for output, result in self.__run_single_step(dataset, params):
                last_output = output
                last_results = result

        if not isinstance(last_output, self.return_type):
            raise TypeError("Provided result type must match promised return_type.")

        return last_output, last_results

    def run_dynamic(self, dataset: alglab.dataset.Dataset, params: Dict):
        if not isinstance(dataset, self.dataset_class):
            raise TypeError("Provided dataset type must match dataset_class expected by the implementation.")

        for param in params.keys():
            if param not in self.all_parameter_names:
                raise ValueError("Unexpected parameter name.")

        if self.number_of_steps > 1:
            raise NotImplementedError("Dynamic multi-step algorithms are not supported.")
        else:
            yield from self.__run_single_step(dataset, params)

    def run(self, dataset: alglab.dataset.Dataset, params: Dict):
        return self.run_static(dataset, params)

    def __repr__(self):
        return self.name
