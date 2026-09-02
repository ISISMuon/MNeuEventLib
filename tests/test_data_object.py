"""
Tests for the `Data` object: loading a file, filtering it, and histogramming.

Unless a test says otherwise, the expected results come from the hand-built
fixture described in `conftest.MockSpec`, where every answer can be worked
out by hand.
"""
import numpy as np
import pytest

from MNeuEventLib.core import Data


class TestConstruction:
    """Loading a file and the state of a `Data` before anything is calculated."""

    def test_repr_reports_the_file_it_loaded(self, mock_data, mock_path):
        assert mock_path in repr(mock_data)

    def test_repr_says_so_before_calculating(self, mock_data):
        assert "result not calculated" in repr(mock_data)

    def test_missing_file_raises(self):
        with pytest.raises(RuntimeError):
            Data("/nonexistent/nope.nxs", n_spec=4)

    def test_split_frame_chunks(self, mock_path, mock):
        """Test we get the right answer when chunk size is smaller than frame size."""
        chunked = Data(mock_path, n_spec=mock.n_spec, chunk_size=5)
        chunked.set_histogram_settings(mock.hist_min, mock.hist_max, mock.hist_bins)
        chunked.calculate()

        np.testing.assert_array_equal(chunked.get_histogram(), mock.expected_hist())


class TestHistogram:

    def test_mock_result(self, mock_data, mock):
        """Compare a calculated result to a mock histogram."""
        mock_data.calculate()
        histogram = mock_data.get_histogram()

        assert histogram.dtype == np.int32
        assert histogram.shape == (mock.n_periods, mock.n_spec, mock.hist_bins)
        np.testing.assert_array_equal(histogram, mock.expected_hist())


class TestRealFile:
    """
    Characterisation tests against the real HIFI run in `tests/test_data`.

    These record what the current implementation produces. They catch
    regressions; they do not show that any of these numbers is correct.
    """

    def test_histogram_shape_and_total(self, real_data):
        real_data.calculate()
        histogram = real_data.get_histogram()

        # characterisation: 2 periods, 64 detectors, the default 2048 bins
        assert histogram.shape == (2, 64, 2048)
        # characterisation: every event in the file lands in the default time range
        assert histogram.sum() == 64147

    @pytest.mark.xfail(
        strict=True,
        reason="Currently miscounted, see issue #93"
    )
    def test_reported_event_count(self, real_data):
        real_data.calculate()

        assert real_data.get_n_events() == 64147

    def test_a_filter_changes_the_result(self, real_data):
        """A time window over part of the run should drop some of the events."""
        real_data.calculate()
        unfiltered = real_data.get_histogram().sum()

        real_data.add_time_filter("first half", 0.0, 1.5)
        real_data.calculate()

        assert 0 < real_data.get_histogram().sum() < unfiltered
