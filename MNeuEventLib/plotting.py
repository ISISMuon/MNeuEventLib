"""Plotting functions for data."""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

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
    ax.set_title(sample_log['name'])
    ax.set_xlabel("Time (seconds)")
    units = sample_log['unit']
    unit_string = "" if units == "" else f"({units})"
    ax.set_ylabel(f"Value {unit_string}")

    values = sample_log['value']
    if isinstance(values, list):
        # a string log comes back as a list of str rather than an array, so
        # we need to plot it differently
        categories = list(dict.fromkeys(values))
        ax.plot(sample_log['time'], values, drawstyle="steps-post", marker="o")
        ax.set_yticks(categories)
        ax.set_ylabel("Value")
    else:
        ax.plot(sample_log['time'], values)

    plt.show()

def plot_amplitudes(data: Data, max_height: Optional[float] = None, n_bins: int = 10):
    """
    Plot a histogram of amplitudes.

    Parameters
    ----------
    data: Data
        The Data object containing the sample log.
    max_height: float, optional
        The maximum height to use.
        Any data above this height will be placed in the last bin.
        If not provided, calculates the maximum amplitude in the data.
    n_bins: int, default 10
        The number of equal-width bins to use.
    """
    # get_amp_histogram returns the histogram and the max height used
    hist, max_h = data.dataset.get_amp_histogram(max_height, n_bins)

    steps = np.linspace(0, max_h, len(hist)+1)
    fig, ax = plt.subplots()
    ax.stairs(hist, steps, fill = True)

    plt.show()
    
