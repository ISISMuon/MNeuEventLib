"""
Shared fixtures for the Python test suite.

These tests drive the library the way a user does: build a NeXuS file, load
it, filter it, and check the numbers that come back.

Most tests use a hand-built fixture file (see `MockSpec`) whose contents are
chosen so that every expected result can be worked out by hand. A few tests
use the real HIFI file in `tests/test_data` and assert values recorded from
the current implementation; those are marked `characterisation:` and prove
only that behaviour has not changed, not that it is right.
"""
import os

# rayon builds its thread pool the first time `calculate` runs, and reads
# RAYON_NUM_THREADS at that point. Setting it here - at conftest import,
# before any test has touched the extension module - is the earliest hook
# pytest gives us, and pinning it keeps float summation order reproducible
# between runs. CI should set it in the environment too, so that the suite
# is still reproducible if this ever stops taking effect.
os.environ.setdefault("RAYON_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # the suite runs headless; must precede any pyplot import

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from MNeuEventLib.core import BatchData, Data  # noqa: E402

REAL_FILE = Path(__file__).parent / "test_data" / "HIFI00195790.nxs"
REAL_N_SPEC = 64


class MockSpec:
    """
    The contents of the hand-built fixture file, and the answers it implies.

    Four frames starting at 0, 1, 2 and 3 seconds, three events in each. The
    three events in a frame belong to detectors 0, 1 and 2 and sit at 0.5,
    1.5 and 2.5 microseconds after the frame start, so with the histogram
    settings below (1 microsecond bins) detector `d`'s event always lands in
    bin `d`. Detector 3 exists but never fires, so its row should stay empty.

    Frames 0 and 1 are period 0; frames 2 and 3 are period 1.
    """

    n_spec = 4
    n_frames = 4
    n_events = 12
    n_periods = 2

    event_id = np.tile([0, 1, 2], n_frames)
    event_time_offset = np.tile([500, 1500, 2500], n_frames)  # ns after frame start
    pulse_height = np.tile([1000.0, 2000.0, 3000.0], n_frames)  # one per detector
    event_index = np.array([0, 3, 6, 9])
    event_time_zero = np.array([0, 1, 2, 3]) * 1_000_000_000  # ns since run start
    period_number = np.array([0, 0, 1, 1])

    # A sample log rising by 10 every second, so that value bounds map onto
    # whole frames: 10 K over frame 0, 20 K over frame 1, and so on.
    log_name = "Temp"
    log_time = np.array([0.0, 1.0, 2.0, 3.0])  # seconds
    log_value = np.array([10.0, 20.0, 30.0, 40.0])
    log_unit = "K"

    # 1 microsecond bins, so bin index == detector index.
    hist_min = 0.0
    hist_max = 4.0
    hist_bins = 4

    def expected_hist(self, frames=(0, 1, 2, 3), detectors=(0, 1, 2)):
        """
        The histogram the fixture should produce for the given kept frames.

        Each kept frame contributes one count at (period, detector, detector)
        for every kept detector.
        """
        hist = np.zeros((self.n_periods, self.n_spec, self.hist_bins), dtype=np.int32)
        for frame in frames:
            for detector in detectors:
                hist[self.period_number[frame], detector, detector] += 1
        return hist

    def reported_n(self, n_events, n_frames):
        """
        What `get_n_events` currently returns for a given number of kept
        events and frames.

        `Histogram::calculate` increments its counter once per kept frame as
        well as once per kept event, so the reported figure is the sum of the
        two rather than the event count alone. Tests that only care about how
        many events survived a filter go through this helper; the discrepancy
        itself is pinned by `test_n_events_counts_events_not_frames`.
        """
        return n_events + n_frames


MOCK = MockSpec()


def write_nexus_file(path, sample_logs=None, **datasets):
    """
    Write a minimal NeXuS event file that the library can read.

    Only the paths the reader actually requires are written. NX_class
    attributes and the run metadata that saving needs are left out
    deliberately: neither is read when loading, so their absence keeps the
    fixture small and makes it obvious what the reader depends on.

    Any of the six event datasets can be replaced by passing it as a keyword
    argument, which is how tests build deliberately malformed files.
    """
    dtypes = {
        "event_id": "u4",
        "event_time_offset": "u4",
        "pulse_height": "f8",
        "event_index": "u4",
        "event_time_zero": "u8",
        "period_number": "u8",
    }
    unexpected = set(datasets) - set(dtypes)
    if unexpected:
        raise TypeError(f"unexpected dataset(s): {sorted(unexpected)}")

    if sample_logs is None:
        sample_logs = {MOCK.log_name: (MOCK.log_time, MOCK.log_value, MOCK.log_unit)}

    with h5py.File(path, "w") as file:
        events = file.create_group("raw_data_1/detector_1_events")
        for name, dtype in dtypes.items():
            default = getattr(MOCK, name)
            events.create_dataset(name, data=np.asarray(datasets.get(name, default), dtype=dtype))

        selog = file.create_group("raw_data_1/selog")
        for name, (time, value, unit) in sample_logs.items():
            log = selog.create_group(f"{name}/value_log")
            log.create_dataset("time", data=np.asarray(time, dtype="f4"))
            values = log.create_dataset("value", data=np.asarray(value, dtype="f4"))
            # a plain str gives a variable-length attribute, which is the only
            # kind the reader can handle
            values.attrs["units"] = unit

    return str(path)


@pytest.fixture
def mock():
    """The specification of the hand-built fixture file."""
    return MOCK


@pytest.fixture
def write_nexus(tmp_path):
    """Factory writing a NeXuS file into this test's temporary directory."""
    counter = iter(range(100))

    def _write(**kwargs):
        return write_nexus_file(tmp_path / f"mock_{next(counter)}.nxs", **kwargs)

    return _write


@pytest.fixture
def mock_path(write_nexus):
    """Path to a fresh copy of the hand-built fixture file."""
    return write_nexus()


@pytest.fixture
def mock_data(mock_path):
    """A `Data` over the hand-built fixture, set up with 1 microsecond bins."""
    data = Data(mock_path, n_spec=MOCK.n_spec)
    data.set_histogram_settings(MOCK.hist_min, MOCK.hist_max, MOCK.hist_bins)
    return data


@pytest.fixture
def make_batch(mock_path):
    """Factory for a `BatchData` over the hand-built fixture."""

    def _make(n_filter_sets=3):
        batch = BatchData(mock_path, n_spec=MOCK.n_spec, n_filter_sets=n_filter_sets)
        batch.set_histogram_settings("all", MOCK.hist_min, MOCK.hist_max, MOCK.hist_bins)
        return batch

    return _make


@pytest.fixture
def real_data():
    """A `Data` over the real HIFI run, used only for characterisation tests."""
    return Data(str(REAL_FILE), n_spec=REAL_N_SPEC)
