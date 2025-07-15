"""Classes and methods related to running experiments on algorithms and datasets."""
import time

from abc import ABC, abstractmethod

from typing import Dict, Type, List, Callable, Union
import itertools
from collections import OrderedDict

import psutil
import os
import threading
import numpy as np

import alglab.dataset
import alglab.results
import alglab.evaluation


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


def resolve_parameters(fixed_parameters, varying_parameters):
    """
    Given a dictionary of fixed parameters and a dictionary of varying parameters, resolve the varying parameters
    which are defined by function.
    """
    resolved_parameters = {}
    dynamic_parameters = []
    for param_name, value in varying_parameters.items():
        if not callable(value):
            # This is a 'static' parameter
            resolved_parameters[param_name] = value
        else:
            # This is a dynamically defined parameter
            dynamic_parameters.append(param_name)

    # Resolve the dynamic parameters
    for param_name in dynamic_parameters:
        resolved_parameters[param_name] = varying_parameters[param_name](fixed_parameters | resolved_parameters)

    return resolved_parameters


class ExperimentSchedule(ABC):
    def __init__(self, dataset: alglab.dataset.Dataset, **kwargs):
        self.dataset = dataset

    @abstractmethod
    def schedule(self):
        pass

    @staticmethod
    @abstractmethod
    def all_method_names():
        pass


class StaticClusteringSchedule(ExperimentSchedule):
    def __init__(self, dataset: alglab.dataset.ClusterableDataset):
        super().__init__(dataset)

    def schedule(self):
        yield 'fit_predict', self.dataset.data, self.dataset.gt_labels

    @staticmethod
    def all_method_names():
        return ['fit_predict']


class FitPredictSchedule(ExperimentSchedule):
    def __init__(self, dataset: alglab.dataset.ClusterableDataset):
        super().__init__(dataset)

    def schedule(self):
        yield 'fit', self.dataset.data, None
        yield 'predict', self.dataset.data, self.dataset.gt_labels

    @staticmethod
    def all_method_names():
        return ['fit', 'predict']


class DynamicClusteringSchedule(ExperimentSchedule):
    def __init__(self, dataset: alglab.dataset.ClusterableDataset, batch_size=1000, stream_by_cluster=False):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.stream_by_cluster = stream_by_cluster

    def schedule(self):
        if self.stream_by_cluster:
            # Stream in the data one cluster at a time
            ordered_indexes = []
            gt_labels = []
            for cluster_id in self.dataset.cluster_ids():
                for id in self.dataset.get_cluster(cluster_id):
                    ordered_indexes.append(id)
                    gt_labels.append(cluster_id)

            reordered_data = self.data[ordered_indexes, :]
        else:
            random_order = np.random.permutation(self.dataset.data.shape[0])
            reordered_data = self.dataset.data[random_order, :]
            gt_labels = self.dataset.gt_labels[random_order]

        current_n = 0
        num_since_last_query = 0
        while current_n < self.dataset.n:
            yield 'add_point', reordered_data[current_n, :], None
            num_since_last_query += 1
            current_n += 1

            if num_since_last_query >= self.batch_size:
                yield 'predict', None, gt_labels[:current_n]
                num_since_last_query = 0

        if num_since_last_query != 0:
            yield 'predict', None, gt_labels[:current_n]

    @staticmethod
    def all_method_names():
        return ['add_point', 'predict']


def monitor_memory(interval, stop_event, result_dict, base_memory):
    process = psutil.Process(os.getpid())
    peak_diff = 0
    while not stop_event.is_set():
        current = process.memory_info().rss
        diff = current - base_memory
        peak_diff = max(peak_diff, diff)
        time.sleep(interval)
    result_dict['peak_diff'] = peak_diff


class Experiment(object):

    def __init__(self, alg: Type,
                 schedule: ExperimentSchedule, params,
                 evaluators = None):
        """An experiment is a single instance of running an algorithm with a set of parameters according
        to some experiment schedule which indicates the order in which algorithm methods should be executed.
        The running time of the algorithm is measured by default. In addition to this, the evaluation_functions
        variable should contain a dictionary of methods which will be applied to the result of the algorithm.
        """
        self.alg = alg
        self.schedule = schedule
        self.params = params
        self.evaluators = evaluators

    def run(self):
        """Run the experiment."""
        # Set up the memory tracking
        this_process = psutil.Process(os.getpid())
        base_memory = this_process.memory_info().rss
        memory_dict = {}
        stop_event = threading.Event()

        # Create the algorithm object
        try:
            alg_obj = self.alg(**self.params)
        except TypeError as e:
            raise TypeError(f"Type error when initialising algorithm {self.alg.__name__}: {e.args[0]}")

        iter = 0
        results_dict = {}
        total_running_times = {}
        iter_running_time = 0
        total_running_time = 0
        for method_name, data, expected_result in self.schedule.schedule():
            # Run the algorithm and measure time and memory usage
            monitor_thread = threading.Thread(
                target=monitor_memory,
                args=(0.1, stop_event, memory_dict, base_memory)
            )
            monitor_thread.start()
            start_time = time.time()
            method_to_run = getattr(alg_obj, method_name)
            if data is None:
                alg_output = method_to_run()
            else:
                alg_output = method_to_run(data)
            end_time = time.time()
            stop_event.set()
            monitor_thread.join()
            peak_memory_bytes = memory_dict.get('peak_diff', 0)
            total_running_time += end_time - start_time
            iter_running_time += end_time - start_time

            iter_step_running_time_key = f'{method_name}_iter_running_time_s'
            iter_total_running_time_key = f'{method_name}_total_running_time_s'
            if iter_step_running_time_key not in results_dict:
                results_dict[iter_step_running_time_key] = end_time - start_time
            else:
                results_dict[iter_step_running_time_key] += end_time - start_time

            if iter_total_running_time_key not in total_running_times:
                total_running_times[iter_total_running_time_key] = end_time - start_time
            else:
                total_running_times[iter_total_running_time_key] += end_time - start_time

            if expected_result is not None:
                # This is the end of an iteration, and we will evaluate
                results_dict |= {
                    'iter': iter,
                    'iter_running_time_s': iter_running_time,
                    'total_running_time_s': total_running_time,
                    'memory_usage_mib': peak_memory_bytes / 1024 / 1024,
                }

                # Apply the evaluation functions
                if self.evaluators:
                    for evaluator in self.evaluators:
                        results_dict[evaluator.__name__] = evaluator(expected_result, alg_output)

                yield results_dict | total_running_times

                iter += 1
                results_dict = {}
                iter_running_time = 0


class ExperimentalSuite(object):

    def __init__(self,
                 algorithms: List[Type],
                 dataset: Union[Type[alglab.dataset.Dataset], alglab.dataset.Dataset],
                 schedule: Type[ExperimentSchedule],
                 results_filename: str,
                 num_runs: int = 1,
                 parameters: Dict = None,
                 evaluators: List[Callable] = None):
        """Run a suite of experiments while varying some parameters.

        Varying parameter dictionaries should have parameter names as keys and the values should be an iterable containing:
            - values to be used directly; or
            - functions, taking fixed and statically defined variable parameters and returning a parameter value
        """
        self.num_runs = num_runs

        if num_runs < 1:
            raise ValueError('num_runs must be greater than or equal to 1')

        self.algorithms = algorithms
        self.algorithm_names = [alg.__name__ for alg in self.algorithms]

        # Automatically populate the parameter dictionaries
        alg_fixed_params = {}
        alg_varying_params = {}
        dataset_fixed_params = {}
        dataset_varying_params = {}
        schedule_fixed_params = {}
        schedule_varying_params = {}

        if parameters is None:
            parameters = {}

        for param_name, param_value in parameters.items():
            alg_dataset_name = ""
            if "." in param_name:
                alg_dataset_name, parameter_name = param_name.split(".")
            else:
                parameter_name = param_name

            if alg_dataset_name == "dataset":
                try:
                    _ = iter(param_value)
                except TypeError:
                    dataset_fixed_params[parameter_name] = param_value
                else:
                    dataset_varying_params[parameter_name] = param_value
            elif alg_dataset_name == "schedule":
                try:
                    _ = iter(param_value)
                except TypeError:
                    schedule_fixed_params[parameter_name] = param_value
                else:
                    schedule_varying_params[parameter_name] = param_value
            elif alg_dataset_name != "":
                try:
                    _ = iter(param_value)
                except TypeError:
                    if alg_dataset_name not in alg_fixed_params:
                        alg_fixed_params[alg_dataset_name] = {}
                    alg_fixed_params[alg_dataset_name][parameter_name] = param_value
                else:
                    if alg_dataset_name not in alg_varying_params:
                        alg_varying_params[alg_dataset_name] = {}
                    alg_varying_params[alg_dataset_name][parameter_name] = param_value
            else:
                for alg_name in self.algorithm_names:
                    try:
                        _ = iter(param_value)
                    except TypeError:
                        if alg_name not in alg_fixed_params:
                            alg_fixed_params[alg_name] = {}
                        alg_fixed_params[alg_name][parameter_name] = param_value
                    else:
                        if alg_name not in alg_varying_params:
                            alg_varying_params[alg_name] = {}
                        alg_varying_params[alg_name][parameter_name] = param_value

        # Check that all the parameters make sense
        for alg in self.algorithms:
            alg_name = alg.__name__

            # Check that every algorithm has an entry in the params dictionary
            if alg_name not in alg_fixed_params:
                alg_fixed_params[alg_name] = {}
            if alg_name not in alg_varying_params:
                alg_varying_params[alg_name] = {}

            # Convert the parameter iterables to lists
            for param_name in alg_varying_params[alg_name].keys():
                alg_varying_params[alg_name][param_name] = list(alg_varying_params[alg_name][param_name])

        # Check that all configured parameters match some algorithm
        for alg in alg_fixed_params.keys():
            if alg not in self.algorithm_names:
                raise ValueError(f"Parameters configured for algorithm {alg} which does not exist.")
        for alg in alg_varying_params.keys():
            if alg not in self.algorithm_names:
                raise ValueError(f"Parameters configured for algorithm {alg} which does not exist.")

        #  Convert parameter iterables to lists
        for param_name in dataset_varying_params.keys():
            dataset_varying_params[param_name] = list(dataset_varying_params[param_name])

        self.alg_fixed_params = alg_fixed_params
        self.alg_varying_params = alg_varying_params
        self.dataset = None
        self.dataset_class = None
        if isinstance(dataset, alglab.dataset.Dataset):
            self.dataset = dataset
        else:
            self.dataset_class = dataset
        self.dataset_fixed_params = dataset_fixed_params
        self.dataset_varying_params = dataset_varying_params

        self.schedule_class = schedule
        self.schedule_fixed_params = schedule_fixed_params
        self.schedule_varying_params = schedule_varying_params

        self.evaluators = evaluators if evaluators is not None else []
        self.results_filename = results_filename

        self.results_columns = self.get_results_df_columns()

        # Compute the total number of experiments to run
        num_datasets = 1
        for param_name, values in self.dataset_varying_params.items():
            num_datasets *= len(values)

        num_schedules = 1
        for param_name, values in self.schedule_varying_params.items():
            num_schedules *= len(values)

        self.num_experiments = 0
        for alg_name in self.algorithm_names:
            num_experiments_this_alg = 1
            for param_name, values in self.alg_varying_params[alg_name].items():
                num_experiments_this_alg *= len(values)
            self.num_experiments += num_experiments_this_alg * num_datasets * num_schedules
        self.num_trials = self.num_experiments * self.num_runs

        self.results = None

    def get_results_df_columns(self):
        """Create a list of all the columns in the results file and dataframe."""
        columns = ['trial_id', 'experiment_id', 'run_id', 'algorithm', 'iter', 'iter_running_time_s', 'total_running_time_s', 'memory_usage_mib']
        for param_name in self.dataset_fixed_params.keys():
            columns.append(param_name)
        for param_name in self.dataset_varying_params.keys():
            columns.append(param_name)
        for param_name in self.schedule_fixed_params.keys():
            columns.append(param_name)
        for param_name in self.schedule_varying_params.keys():
            columns.append(param_name)

        for alg_name in self.algorithm_names:
            for param_name in self.alg_fixed_params[alg_name].keys():
                columns.append(param_name)
            for param_name in self.alg_varying_params[alg_name].keys():
                columns.append(param_name)

        for method_name in self.schedule_class.all_method_names():
            columns.append(f'{method_name}_iter_running_time_s')
            columns.append(f'{method_name}_total_running_time_s')

        for evaluator in self.evaluators:
            columns.append(evaluator.__name__)

        return list(OrderedDict.fromkeys(columns))

    def run_all(self, append_results=False) -> alglab.results.Results:
        """Run all the experiments in this suite."""

        # If we are appending the results, make sure that the header of the results file already matches the
        # header we would have written.
        if append_results:
            existing_results = alglab.results.Results(self.results_filename)
            if existing_results.column_names() != self.results_columns:
                raise ValueError("Cannot append results file: column names do not match.")
            true_trial_number = existing_results.results_df.iloc[-1]["trial_id"] + 1
            base_experiment_number = existing_results.results_df.iloc[-1]["experiment_id"] + 1
        else:
            true_trial_number = 1
            base_experiment_number = 1

        reported_trial_number = 1

        file_access_string = 'a' if append_results else 'w'

        with open(self.results_filename, file_access_string) as results_file:
            # Write the header line of the results file
            if not append_results:
                results_file.write(", ".join(self.results_columns))
                results_file.write("\n")

            for run in range(1, self.num_runs + 1):
                experiment_number = base_experiment_number
                for dataset_params in product_dict(**self.dataset_varying_params):
                    resolved_varying_dataset_params = resolve_parameters(self.dataset_fixed_params, dataset_params)
                    full_dataset_params = self.dataset_fixed_params | resolved_varying_dataset_params
                    dataset = self.dataset
                    if self.dataset is None:
                        dataset = self.dataset_class(**full_dataset_params)

                    for schedule_params in product_dict(**self.schedule_varying_params):
                        resolved_schedule_params = resolve_parameters(self.schedule_fixed_params, schedule_params)
                        full_schedule_params = self.schedule_fixed_params | resolved_schedule_params
                        schedule = self.schedule_class(dataset, **full_schedule_params)

                        for alg in self.algorithms:
                            alg_name = alg.__name__
                            for alg_params in product_dict(**self.alg_varying_params[alg_name]):
                                resolved_varying_alg_params = resolve_parameters(full_dataset_params | self.alg_fixed_params[alg_name], alg_params)
                                full_alg_params = self.alg_fixed_params[alg_name] | resolved_varying_alg_params
                                print(f"Trial {reported_trial_number} / {self.num_trials}: {alg_name} on {dataset} with parameters {full_alg_params}")
                                this_experiment = Experiment(alg, schedule, full_alg_params, self.evaluators)

                                for result in this_experiment.run():
                                    this_result = result | full_dataset_params | full_alg_params | full_schedule_params | \
                                                  {'algorithm': alg_name, 'trial_id': true_trial_number,
                                                   'experiment_id': experiment_number, 'run_id': run}
                                    results_file.write(", ".join([str(this_result[col]) if col in this_result else '' for col in self.results_columns]))
                                    results_file.write("\n")
                                    results_file.flush()

                                true_trial_number += 1
                                reported_trial_number += 1
                                experiment_number += 1

        # Create a dataframe from the results
        self.results = alglab.results.Results(self.results_filename)
        return self.results
