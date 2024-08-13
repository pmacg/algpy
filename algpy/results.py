"""Classes and method for processing experimental results."""
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


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

    def line_plot(self, x_col, y_col, filename=None):
        """Plot one column of the dataframe against another."""
        fig, ax = plt.subplots(figsize=(4, 3))
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)

        for alg_name in self.algorithm_names:
            this_alg_results = self.stats_df[(self.stats_df['algorithm'] == alg_name)]
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
        plt.show()

