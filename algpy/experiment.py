"""Classes and methods related to running experiments on algorithms and datasets."""
import time

from typing import Dict, Type, List, Callable, Iterable
import pandas as pd
import itertools
from collections import OrderedDict

import algpy.algorithm
import algpy.dataset
import algpy.results


def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


class Experiment(object):

    def __init__(self, alg: algpy.algorithm.Algorithm, dataset: algpy.dataset.Dataset, params,
                 evaluation_functions=None):
        """An experiment is a single instance of running an algorithm on a dataset with a set of parameters.
        The running time of the algorithm is measured by default. In addition to this, the evaluation_functions
        variable should contain a dictionary of methods which will be applied to the result of the algorithm.
        """
        self.alg = alg
        self.dataset = dataset
        self.params = params
        self.evaluation_functions = evaluation_functions

        # The result will be a dictionary of information we want to track after each run.
        self.result = {}

    def run(self):
        """Run the experiment."""
        # We always measure the running time of the experiment.
        start_time = time.time()
        alg_output = self.alg.run(self.dataset, self.params)
        end_time = time.time()
        self.result['running_time_s'] = end_time - start_time

        # Apply the evaluation functions
        if self.evaluation_functions:
            for metric, func in self.evaluation_functions.items():
                self.result[metric] = func(self.dataset, alg_output)

class ExperimentalSuite(object):

    def __init__(self,
                 algorithms: Dict[str, Callable],
                 dataset: Type[algpy.dataset.Dataset],
                 results_filename: str,
                 alg_fixed_params: Dict[str, Dict]=None,
                 alg_varying_params:  Dict[str, Dict[str, Iterable]] = None,
                 dataset_fixed_params: Dict = None,
                 dataset_varying_params: Dict[str, Iterable] = None,
                 evaluation_functions: Dict[str, Callable] = None):
        """Run a suite of experiments while varying some parameters."""

        self.algorithms = {name: algpy.algorithm.Algorithm(func) for name, func in algorithms.items()}

        # Automatically populate the parameter dictionaries
        if alg_fixed_params is None:
            alg_fixed_params = {}
        if alg_varying_params is None:
            alg_varying_params = {}
        if dataset_fixed_params is None:
            dataset_fixed_params = {}
        if dataset_varying_params is None:
            dataset_varying_params = {}

        for alg_name in self.algorithms.keys():
            # Check that every algorithm has an entry in the params dictionary
            if alg_name not in alg_fixed_params:
                alg_fixed_params[alg_name] = {}
            if alg_name not in alg_varying_params:
                alg_varying_params[alg_name] = {}

            # Convert the parameter iterables to lists
            for param_name in alg_varying_params[alg_name].keys():
                alg_varying_params[alg_name][param_name] = list(alg_varying_params[alg_name][param_name])

        #  Convert parameter iterables to lists
        for param_name in dataset_varying_params.keys():
            dataset_varying_params[param_name] = list(dataset_varying_params[param_name])

        self.alg_fixed_params = alg_fixed_params
        self.alg_varying_params = alg_varying_params
        self.dataset_class = dataset
        self.dataset_fixed_params = dataset_fixed_params
        self.dataset_varying_params = dataset_varying_params
        self.evaluation_functions = evaluation_functions
        self.results_filename = results_filename

        self.results_columns = self.get_results_df_columns()

        # Compute the total number of experiments to run
        num_datasets = 1
        for param_name, values in self.dataset_varying_params.items():
            num_datasets *= len(values)
        self.num_experiments = 0
        for alg_name in self.algorithms.keys():
            num_experiments_this_alg = 1
            for param_name, values in self.alg_varying_params[alg_name].items():
                num_experiments_this_alg *= len(values)
            self.num_experiments += num_experiments_this_alg * num_datasets

        self.results = None

    def get_results_df_columns(self):
        """Create a list of all the columns in the results file and dataframe."""
        columns = ['trial_id', 'algorithm']
        for param_name in self.dataset_fixed_params.keys():
            columns.append(param_name)
        for param_name in self.dataset_varying_params.keys():
            columns.append(param_name)
        for alg_name in self.algorithms.keys():
            for param_name in self.alg_fixed_params[alg_name].keys():
                columns.append(param_name)
            for param_name in self.alg_varying_params[alg_name].keys():
                columns.append(param_name)
        columns.append('running_time_s')
        for eval_name in self.evaluation_functions.keys():
            columns.append(eval_name)
        return list(OrderedDict.fromkeys(columns))

    def run_all(self, append_results=False) -> algpy.results.Results:
        """Run all the experiments in this suite."""

        # If we are appending the results, make sure that the header of the results file already matches the
        # header we would have written.
        if append_results:
            existing_results = algpy.results.Results(self.results_filename)
            if existing_results.column_names() != self.results_columns:
                raise ValueError("Cannot append results file: column names do not match.")
            true_trial_number = existing_results.results_df.iloc[-1]["trial_id"] + 1
        else:
            true_trial_number = 1

        reported_trial_number = 1

        file_access_string = 'a' if append_results else 'w'

        with open(self.results_filename, file_access_string) as results_file:
            # Write the header line of the results file
            if not append_results:
                results_file.write(", ".join(self.results_columns))
                results_file.write("\n")

            for dataset_params in product_dict(**self.dataset_varying_params):
                full_dataset_params = self.dataset_fixed_params | dataset_params
                dataset = self.dataset_class(**full_dataset_params)

                for alg_name, alg in self.algorithms.items():
                    for alg_params in product_dict(**self.alg_varying_params[alg_name]):
                        full_alg_params = self.alg_fixed_params[alg_name] | alg_params
                        print(f"Trial {reported_trial_number} / {self.num_experiments}: {alg_name} on {dataset} with parameters {full_alg_params}.")
                        this_experiment = Experiment(alg, dataset, full_alg_params, self.evaluation_functions)
                        this_experiment.run()

                        this_result = this_experiment.result | full_dataset_params | full_alg_params | {'algorithm': alg_name, 'trial_id': true_trial_number}
                        results_file.write(", ".join([str(this_result[col]) if col in this_result else '' for col in self.results_columns]))
                        results_file.write("\n")
                        results_file.flush()

                        true_trial_number += 1
                        reported_trial_number += 1

        # Create a dataframe from the results
        self.results = algpy.results.Results(self.results_filename)
        return self.results
