"""Plotting functions for data."""

import matplotlib.pyplot as plt

from MNeuEventLib.core import Data

def plot_sample_log(data: Data, log_name: str):
    """
    Plot a sample log.

    Parameters
    ----------
    data: Data
        The Data object containing the sample log.
    log_name: str
        The name of the sample log to plot.
    """
    sample_log = data.dataset.get_sample_log(log_name)
    fig, ax = plt.subplots()
    ax.plot(sample_log['time'], sample_log['value'])

    plt.show()
