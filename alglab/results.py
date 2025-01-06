"""Classes and method for processing experimental results."""
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np


class Results(object):

    def __init__(self, results_filename):
        """Load results from a csv file."""
        self.results_df = pd.read_csv(results_filename, skipinitialspace=True)
        self.algorithm_names = self.results_df['algorithm'].unique()
        self.num_runs = self.results_df['run_id'].max()
        self.stats_df = self.create_averaged_stats()

    def create_averaged_stats(self):
        """
        Create averages and error bars from multiple runs.
        """
        alg_dfs = {alg_name: self.results_df.loc[self.results_df['algorithm'] == alg_name] for alg_name in
                  self.algorithm_names}
        stats_df = pd.DataFrame()
        for alg_name, alg_df in alg_dfs.items():
            mean_df = alg_df.groupby(['experiment_id']).mean(numeric_only=True)
            mean_df = mean_df.drop(['trial_id', 'run_id'], axis=1).add_prefix('_mean_')
            sem_df = alg_df.groupby(['experiment_id']).sem(numeric_only=True)
            sem_df = sem_df.drop(['trial_id', 'run_id'], axis=1).add_prefix('_sem_')
            sem_df['algorithm'] = alg_name
            sem_df = sem_df.join(mean_df)
            stats_df = pd.concat([stats_df, sem_df])
        return stats_df

    def column_names(self) -> List[str]:
        return self.results_df.columns.values.tolist()

    def line_plot(self, x_col, y_col, filename=None,
                  ignore_algorithms=None,
                  fixed_parameters=None):
        """Plot one column of the dataframe against another."""
        if ignore_algorithms is None:
            ignore_algorithms = []

        if fixed_parameters is None:
            fixed_parameters = {}

        # Filter all the results by the provided fixed parameters
        results_to_plot = self.stats_df
        for param, val in fixed_parameters.items():
            results_to_plot = results_to_plot[results_to_plot[f'_mean_{param}'] == val]

        fig, ax = plt.subplots(figsize=(4, 3))
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)

        for alg_name in self.algorithm_names:
            if alg_name not in ignore_algorithms:
                this_alg_results = results_to_plot[(self.stats_df['algorithm'] == alg_name)]
                plt.plot(this_alg_results[f"_mean_{x_col}"],
                         this_alg_results[f"_mean_{y_col}"],
                         linewidth=3,
                         label=alg_name)
                if self.num_runs > 1:
                    plt.fill_between(this_alg_results[f"_mean_{x_col}"],
                                     this_alg_results[f"_mean_{y_col}"] - this_alg_results[f"_sem_{y_col}"],
                                     this_alg_results[f"_mean_{y_col}"] + this_alg_results[f"_sem_{y_col}"],
                                     alpha=0.2)

        plt.legend()
        if filename:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.tight_layout()
        plt.show()

    def bar_plot(self, x_col, y_col, x_vals, filename=None,
                 ignore_algorithms=None,
                 fixed_parameters=None):
        if ignore_algorithms is None:
            ignore_algorithms = []

        if fixed_parameters is None:
            fixed_parameters = {}

        # Filter all the results by the provided fixed parameters
        results_to_plot = self.stats_df
        for param, val in fixed_parameters.items():
            results_to_plot = results_to_plot[results_to_plot[param] == val]

        x = np.arange(len(x_vals))  # the label locations
        width = 0.75 / (len(self.algorithm_names) - len(ignore_algorithms))  # the width of the bars

        fig, ax = plt.subplots(layout='constrained')

        multiplier = 0
        for alg_name in self.algorithm_names:
            if alg_name not in ignore_algorithms:
                offset = width * multiplier
                alg_results = results_to_plot[(self.stats_df['algorithm'] == alg_name)]
                y_vals = []
                for x_val in x_vals:
                    y_vals.append(alg_results[alg_results[f"_mean_{x_col}"] == x_val][f"_mean_{y_col}"].values[0])
                rects = ax.bar(x + offset, y_vals, width, label=alg_name)
                ax.bar_label(rects, padding=3)
                multiplier += 1

        plt.legend()
        if filename:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.tight_layout()
        plt.show()

