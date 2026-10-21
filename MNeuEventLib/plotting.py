"""Plotting functions for data."""
import warnings
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from MNeuEventLib.core import Data

# matplotlib's default colour cycle, which we use while it has enough colours
_N_CYCLE_COLOURS = 10

def plot_sample_log(data: Data, log_name: str):
    """
    Plot a sample log.

    Parameters
    ----------
    data: Data
        The Data object containing the sample log.
    log_name: str
        The name of the sample log to plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure that was plotted.
    matplotlib.axes.Axes
        The axes that were plotted on.
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

    return fig, ax

def _normalise_detectors(detector: Optional[Union[int, str, Sequence[int]]], n_spec: int):
    """
    Work out which detectors to plot from the `detector` argument.

    Returns None to histogram every detector summed together, an int to plot a single
    detector, or a list of at least two detectors to overlay. A sequence of one
    detector is just a single detector, as there is nothing to compare it to.
    """
    if detector is None or isinstance(detector, (int, np.integer)):
        return detector

    # a str is a sequence, so this has to be checked before we try to read
    # the argument as a sequence of detector numbers
    if isinstance(detector, str):
        if detector.lower() != "all":
            raise ValueError(
                f"Unknown detector selection '{detector}'. Give a detector number, "
                "a sequence of detector numbers, or 'all' to overlay every detector."
            )
        detector = range(n_spec)

    detectors = [int(number) for number in detector]
    if not detectors:
        raise ValueError(
            "No detectors were given to plot. To plot all detectors "
            "together, leave the `detector` argument unset."
        )
    if len(detectors) == 1:
        return detectors[0]
    return detectors

def _detector_colours(n_detectors: int):
    """
    Get a visually distinct colour for each detector.
    """
    # matplotlib's default colour cycle only has ten colours and wraps around
    # so for > 10 detectors we need to grab intervals from a continuous colour cycle instead
    if n_detectors <= _N_CYCLE_COLOURS:
        return [f"C{i}" for i in range(n_detectors)]
    return plt.get_cmap("viridis")(np.linspace(0, 1, n_detectors))

def plot_amplitudes(data: Data, max_height: Optional[float] = None, n_bins: int = 10,
                    detector: Optional[Union[int, str, Sequence[int]]] = None):
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
    detector: int, sequence of int, or "all", optional
        The detector, or detectors, to plot amplitudes for.
        If several detectors are given, one histogram per detector is
        overlaid so that they can be compared. Pass "all" to overlay
        every detector on the instrument.
        If not provided, plots all detectors together as a single histogram.

    Returns
    -------
    matplotlib.figure.Figure
        The figure that was plotted.
    matplotlib.axes.Axes
        The axes that were plotted on.
    """
    detectors = _normalise_detectors(detector, data.dataset.n_spec)
    overlay = isinstance(detectors, list)

    # both functions return the histogram(s) and the max height used.
    # every detector is binned over the same edges either way, so the
    # histograms can be compared directly
    if overlay:
        hists, max_h = data.dataset.get_amp_histograms(detectors, max_height, n_bins)
    else:
        hist, max_h = data.dataset.get_amp_histogram(max_height, n_bins, detectors)
        hists = [hist]

    fig, ax = plt.subplots()
    # with several detectors the legend says which is which
    if overlay or detectors is None:
        ax.set_title("Amplitude histogram")
    else:
        ax.set_title(f"Amplitude histogram (detector {detectors})")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Counts")

    # every detector shares the same bin edges, so they share an x-axis
    steps = np.linspace(0, max_h, len(hists[0])+1)
    if overlay:
        # draw multiple overlaid hystograms transparent so they can all be seen
        colours = _detector_colours(len(detectors))
        for number, hist, colour in zip(detectors, hists, colours):
            ax.stairs(hist, steps, fill = True, alpha = 0.4, color = colour,
                      label = f"Detector {number}")
        ax.legend()
    else:
        ax.stairs(hists[0], steps, fill = True)

    # if the detector plotted is completely empty, warn the user so they know it's missing 
    if detectors is not None:
        numbers = detectors if overlay else [detectors]
        empty = [number for number, hist in zip(numbers, hists) if not np.any(hist)]
        if len(empty) == 1:
            warnings.warn(f"No events were recorded by detector {empty[0]}, "
                          "so it is plotted as an empty histogram.")
        elif empty:
            warnings.warn("No events were recorded by detectors "
                          + ", ".join(str(number) for number in empty)
                          + ", so they are plotted as empty histograms.")

    plt.show()

    return fig, ax

