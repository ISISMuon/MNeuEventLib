"""
Tests for `BatchData`.

The filtering behaviour itself is covered in `test_data_object.py`; what
matters here is that each filter set is independent, and that the leading
index argument accepts what it should and rejects what it should not.
"""
import numpy as np
import pytest

class TestConstruction:
    """Creating a batch and inspecting it before anything is calculated."""

    @pytest.mark.parametrize("n_filter_sets", [1, 2, 5])
    def test_len_is_the_number_of_filter_sets(self, make_batch, n_filter_sets):
        assert len(make_batch(n_filter_sets)) == n_filter_sets

    def test_repr_describes_every_filter_set(self, make_batch):
        text = repr(make_batch(3))

        for index in range(3):
            assert f"Filter set {index}" in text

class TestIndexing:
    """Test indexing."""

    def test_filter_index(self, make_batch, mock):
        """Test indexing with a number only adds to one filter set."""
        batch = make_batch(3)
        batch.add_time_filter(1, "window", 1.0, 2.0)
        batch.calculate()
        histograms = batch.get_histogram("all")

        np.testing.assert_array_equal(histograms[0], mock.expected_hist())
        np.testing.assert_array_equal(histograms[1], mock.expected_hist(frames=(1, 2)))
        np.testing.assert_array_equal(histograms[2], mock.expected_hist())

    def test_filter_all(self, make_batch, mock):
        """Test indexing with 'all' adds to all filter sets."""
        batch = make_batch(3)
        batch.add_time_filter('all', "window", 1.0, 2.0)
        batch.calculate()
        histograms = batch.get_histogram("all")

        np.testing.assert_array_equal(histograms[0], mock.expected_hist(frames=(1, 2)))
        np.testing.assert_array_equal(histograms[1], mock.expected_hist(frames=(1, 2)))
        np.testing.assert_array_equal(histograms[2], mock.expected_hist(frames=(1, 2)))

