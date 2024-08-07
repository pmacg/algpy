"""Classes and method for processing experimental results."""
import pandas as pd
import matplotlib.pyplot as plt


class Results(object):

    def __init__(self, results_filename):
        """Load results from a csv file."""
        self.results_df = pd.read_csv(results_filename, skipinitialspace=True)
        self.algorithm_names = self.results_df['algorithm'].unique()

    def line_plot(self, x_col, y_col, filename=None):
        """Plot one column of the dataframe against another."""
        plt.figure(figsize=(4, 3))
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)

        for alg_name in self.algorithm_names:
            this_alg_results = self.results_df[(self.results_df['algorithm'] == alg_name)]
            plt.plot(this_alg_results[x_col],
                     this_alg_results[y_col],
                     linewidth=3,
                     label=alg_name)

        plt.legend()
        if filename:
            plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.show()

