"""
Tests for `MNeuEventLib.plotting`.

These are smoke tests: they check that each function draws what it says it
draws, not that the picture looks right. The backend is forced to Agg in
`conftest.py`, so `plt.show()` is a no-op and the figure stays available.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from MNeuEventLib.plotting import plot_amplitudes, plot_sample_log


@pytest.fixture(autouse=True)
def headless_figures():
    """
    Keep the plotting functions quiet and tidy.

    Both of them end in `plt.show()`, which warns under Agg; that is expected
    when running headless rather than anything the tests should report. The
    figure itself survives `show`, so it can be inspected afterwards.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        yield
    plt.close("all")


class TestPlotSampleLog:
    def test_plots_the_log_values_against_time(self, mock_data, mock):
        plot_sample_log(mock_data, mock.log_name)
        (line,) = plt.gcf().axes[0].lines

        np.testing.assert_allclose(line.get_xdata(), mock.log_time)
        np.testing.assert_allclose(line.get_ydata(), mock.log_value)

    def test_titles_the_plot_with_the_log_name(self, mock_data, mock):
        plot_sample_log(mock_data, mock.log_name)

        assert plt.gcf().axes[0].get_title() == mock.log_name

    def test_labels_the_time_axis(self, mock_data, mock):
        plot_sample_log(mock_data, mock.log_name)

        assert plt.gcf().axes[0].get_xlabel() == "Time (seconds)"

    @pytest.mark.xfail(
        strict=True,
        reason="the walrus in plot_sample_log binds the result of the comparison "
        "rather than the unit, so the label reads 'Value (False)'. " 
        "See issue #107",
    )
    def test_labels_the_value_axis_with_the_unit(self, mock_data, mock):
        plot_sample_log(mock_data, mock.log_name)

        assert plt.gcf().axes[0].get_ylabel() == f"Value ({mock.log_unit})"


class TestPlotAmplitudes:
    def test_draws_a_histogram(self, mock_data):
        plot_amplitudes(mock_data)

        assert len(plt.gcf().axes[0].patches) == 1

    def test_uses_the_requested_number_of_bins(self, mock_data):
        plot_amplitudes(mock_data, n_bins=5)
        (stairs,) = plt.gcf().axes[0].patches

        assert len(stairs.get_data().values) == 5

    def test_a_max_height_bounds_the_last_edge(self, mock_data):
        plot_amplitudes(mock_data, max_height=2000.0, n_bins=4)
        (stairs,) = plt.gcf().axes[0].patches

        assert stairs.get_data().edges[-1] == pytest.approx(2000.0)
