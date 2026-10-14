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
    if (units := sample_log['unit']) == "":
        unit_string = ""
    else:
        unit_string = f"({units})"
    ax.set_ylabel(f"Value {unit_string}")
    ax.plot(sample_log['time'], sample_log['value'])

    plt.show()

def plot_amplitudes(data: Data, max_height: Optional[float] = None, n_bins: int = 10,
                    detector: Optional[int] = None):
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
    detector: int, optional
        The detector to plot amplitudes for.
        If not provided, plots for all detectors.
    """
    # get_amp_histogram returns the histogram and the max height used
    hist, max_h = data.dataset.get_amp_histogram(max_height, n_bins, detector)

    steps = np.linspace(0, max_h, len(hist)+1)
    fig, ax = plt.subplots()
    if detector is None:
        ax.set_title("Amplitude histogram")
    else:
        ax.set_title(f"Amplitude histogram (detector {detector})")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Counts")
    ax.stairs(hist, steps, fill = True)

    plt.show()
    
