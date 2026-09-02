use std::cmp::min;
use std::iter::Iterator;
use std::sync::Mutex;

use anyhow::{Error, Result};
use ndarray::{s, Array1, Array3};
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, Bound};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::consts::ToMicroseconds;
use crate::data::{FrameData, NexusData};
use crate::filters::{get_weights, Filters, Weights};

type PyHist<'py> = Bound<'py, PyArray3<i32>>;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Histogram {
    pub min_time: u32,
    pub max_time: u32,
    pub n_bins: usize,
    pub hist: Array3<i32>,
    pub n: usize,
    pub n_frames: Vec<u32>,
    pub n_good_frames: Vec<u32>,
    pub start_time: usize,
    pub end_time: usize,
}

/// Per-worker accumulation state for `calculate_histograms`.
///
/// this holds the working data as we build up the histogram
/// for each thread, then they're all combined at the end
struct Accumulator {
    hist: Array3<i32>,
    n: usize,
}

impl Accumulator {
    fn zeros(n_periods: usize, n_spec: usize, n_bins: usize) -> Accumulator {
        Accumulator {
            hist: Array3::zeros((n_periods, n_spec, n_bins)),
            n: 0,
        }
    }

    fn merge(&mut self, other: &Accumulator) {
        self.hist += &other.hist;
        self.n += other.n;
    }
}

#[pymethods]
impl Histogram {
    fn data<'py>(slf: &Bound<'py, Histogram>) -> PyHist<'py> {
        let py = slf.py();
        slf.borrow().hist.to_pyarray(py)
    }

    fn n_events(&self) -> usize {
        self.n
    }

    pub fn __repr__(&self) -> String {
        let shape = self.hist.shape();
        let mut string = format!(
            "Histogram with:\n  time range {}μs - {}μs",
            (self.min_time.to_micros()),
            (self.max_time.to_micros())
        );
        if shape == [0, 0, 0] {
            string += &format!("\n  {} bins\n  result not calculated", self.n_bins);
        } else {
            let plural_periods = if shape[0] > 1 { "s" } else { "" };
            string += &format!(
                "\n  {} period{}\n  {} detectors\n  {} bins\n  {} events",
                shape[0], plural_periods, shape[1], shape[2], self.n
            )
        }
        string
    }
}

impl Histogram {
    pub fn new(min_time: u32, max_time: u32, n_bins: usize) -> Histogram {
        // note min_time and max_time are given as microseconds
        Histogram {
            min_time,
            max_time,
            n_bins,
            hist: Array3::zeros((0, 0, 0)),
            n: 0,
            n_frames: vec![0],
            n_good_frames: vec![0],
            start_time: 0,
            end_time: 0,
        }
    }

    pub fn calculate(&self, data: &NexusData, filters: &Filters) -> Result<Histogram> {
        // get period data
        let periods: Array1<u32> = data.periods.read_1d()?;
        let n_periods = (periods.iter().max().unwrap() + 1) as usize;

        // set up data to parse things recorded by frame rather than by event
        let start_index: Array1<usize> = data.frames.read_1d()?;
        let frame_data = FrameData::new(start_index.clone(), data.n_events);

        // get data for time filters
        let (time_starts, time_ends) = filters.get_time_filter_times();

        // get data for all log filters that have been filtered
        let log_names = filters.get_required_log_names();
        let value_logs = match data.get_sample_logs(log_names) {
            Ok(logs) => logs,
            Err(info) => return Err(Error::msg(format!("Failed to get logs: {info}"))),
        };
        let (log_starts, log_ends) = filters.get_log_filter_times(value_logs);

        let filters_exist = !time_starts.is_empty() || !log_starts.is_empty();

        let frame_start_times: Array1<usize> = data.frame_times.read_1d()?;

        let weights = if filters_exist {
            let time_weights = if time_starts.is_empty() {
                Weights::ones(data.n_frames)
            } else {
                get_weights(
                    time_starts,
                    time_ends,
                    &frame_start_times,
                    filters.is_include(),
                )
            };
            // log weights are always include filters
            let log_weights = if log_starts.is_empty() {
                Weights::ones(data.n_frames)
            } else {
                get_weights(log_starts, log_ends, &frame_start_times, true)
            };
            time_weights & log_weights
        } else {
            Weights::ones(data.n_frames)
        };

        // todo: once vetos are added, calculate n_good_frames too
        let n_frames = if n_periods == 1 {
            vec![weights.count()]
        } else {
            get_period_frames(&periods, n_periods, &weights)
        };

        let min_amps = filters.get_amps(data.n_spec)?;

        let mut histogram = calculate_histograms(
            data,
            self.min_time,
            self.max_time,
            self.n_bins,
            n_periods,
            periods,
            min_amps,
            &weights,
            frame_data,
        );
        histogram.n_frames = n_frames.clone();
        histogram.n_good_frames = n_frames;

        (histogram.start_time, histogram.end_time) =
            get_experiment_times(weights, frame_start_times);

        Ok(histogram)
    }
}

/// Calculate the number of kept frames for each period.
pub fn get_period_frames(periods: &Array1<u32>, n_periods: usize, weights: &Weights) -> Vec<u32> {
    let mut output = vec![0; n_periods];
    for (k, period) in periods.iter().enumerate() {
        if weights[k] {
            output[*period as usize] += 1
        }
    }
    output
}

/// Get the start and end times of the (optionally filtered) experiment.
pub fn get_experiment_times(weights: Weights, frame_start_times: Array1<usize>) -> (usize, usize) {
    (
        frame_start_times[weights.get_first_one().unwrap()],
        frame_start_times[weights.get_last_one().unwrap()],
    )
}

/// Calculate histograms and output the result.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn calculate_histograms(
    dataset: &NexusData,
    min_time: u32,
    max_time: u32,
    n_bins: usize,
    n_periods: usize,
    periods: Array1<u32>,
    min_amps: Array1<f64>,
    weights: &Weights,
    frame_data: FrameData,
) -> Histogram {
    let width: f32 = (max_time - min_time) as f32 / n_bins as f32;

    // accumulate one histogram per worker
    // current_thread_index may be None (if it runs on a thread that isn't a Rayon thread somehow)
    // so we keep a fallback slot just in case
    let n_threads = rayon::current_num_threads() + 1;
    let fallback_slot = n_threads - 1;
    let accumulators: Vec<Mutex<Option<Accumulator>>> =
        (0..n_threads).map(|_| Mutex::new(None)).collect();

    // iterate over the data chunks, make histograms for each, then sum histograms at the end
    (0..dataset.n_events)
        .into_par_iter()
        .step_by(dataset.chunk_size)
        .for_each(|start| {
            let end = min(start + (dataset.chunk_size), dataset.n_events);
            let array_slice = s![start..end];
            let amps: Array1<f64> = dataset
                .amps
                .read_slice_1d(array_slice)
                .expect("failed to read amplitudes.");
            let times: Array1<u32> = dataset
                .times
                .read_slice_1d(array_slice)
                .expect("Failed to read times.");
            let specs: Array1<u32> = dataset
                .specs
                .read_slice_1d(array_slice)
                .expect("Failed to read specs.");

            let thread = rayon::current_thread_index().unwrap_or(fallback_slot);
            let mut acc_mutex = accumulators[thread]
                .lock()
                .expect("accumulator thread panicked!");
            let acc = acc_mutex
                .get_or_insert_with(|| Accumulator::zeros(n_periods, dataset.n_spec, n_bins));

            make_histogram(
                acc,
                times,
                specs,
                amps,
                &periods,
                &min_amps,
                weights,
                frame_data.slice(start, end),
                min_time,
                max_time,
                width,
            );
        });

    // merge accumulators
    let result = accumulators
        .into_iter()
        .filter_map(|thread| thread.into_inner().expect("accumulator thread panicked!"))
        .reduce(|mut acc, other| {
            acc.merge(&other);
            acc
        })
        .unwrap_or_else(|| Accumulator::zeros(n_periods, dataset.n_spec, n_bins));

    let mut histogram = Histogram::new(min_time, max_time, n_bins);
    histogram.hist = result.hist;
    histogram.n = result.n;
    histogram
}

/// Make a histogram for a set of data into `acc`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn make_histogram(
    acc: &mut Accumulator,
    times: Array1<u32>,
    specs: Array1<u32>,
    amps: Array1<f64>,
    periods: &Array1<u32>,
    min_amps: &Array1<f64>,
    weights: &Weights,
    frame_data: FrameData,
    min_time: u32,
    max_time: u32,
    width: f32,
) {
    let last_frame = frame_data.frame_number.last().unwrap();

    // iterate over the frames in the slice
    for (i, frame) in frame_data.frame_number.iter().enumerate() {
        // if the weight for this frame is 0, skip the frame
        if !weights[*frame] {
            continue;
        }

        // get event indices of this frame in the slice
        let frame_start_event = frame_data.start_index[i];
        let frame_end_event = if frame == last_frame {
            frame_data.array_len
        } else {
            frame_data.start_index[i + 1]
        };

        let period = periods[*frame] as usize;

        for k in frame_start_event..frame_end_event {
            let t = times[k];
            let amp = amps[k];
            let spec = specs[k] as usize;

            if (t >= min_time) && (t < max_time) && amp > min_amps[spec] {
                let bin = ((t - min_time) as f32 / width) as usize;
                acc.hist[[period, spec, bin]] += 1;
                acc.n += 1
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture event file already used by other test modules in the crate
    /// (see `data::nexus_data::tests`).
    const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";

    /// Allocate an accumulator, run one chunk of data into it, and return it.
    /// Keeps the pre-accumulator call shape for tests that check a single chunk.
    #[allow(clippy::too_many_arguments)]
    fn histogram_one_chunk(
        times: Array1<u32>,
        specs: Array1<u32>,
        amps: Array1<f64>,
        n_spec: usize,
        periods: &Array1<u32>,
        n_periods: usize,
        min_amps: &Array1<f64>,
        weights: &Weights,
        frame_data: FrameData,
        min_time: u32,
        max_time: u32,
        n_bins: usize,
        width: f32,
    ) -> Accumulator {
        let mut acc = Accumulator::zeros(n_periods, n_spec, n_bins);
        make_histogram(
            &mut acc, times, specs, amps, periods, min_amps, weights, frame_data, min_time,
            max_time, width,
        );
        acc
    }

    /// Test Histogram::new creates correct empty histogram.
    #[test]
    fn test_histogram_new() {
        let hist = Histogram::new(500, 2500, 4);
        assert_eq!(hist.min_time, 500);
        assert_eq!(hist.max_time, 2500);
        assert_eq!(hist.n_bins, 4);
        assert_eq!(hist.n, 0);
        assert_eq!(hist.hist.dim(), (0, 0, 0));
    }

    /// Test Histogram::n_events returns correct count.
    #[test]
    fn test_histogram_n_events() {
        let mut hist = Histogram::new(0, 3000, 3);
        hist.n = 42;
        assert_eq!(hist.n_events(), 42);
    }

    /// Test a histogram with no filters is correctly constructed.
    #[test]
    fn test_hist_no_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let amps = Array1::ones(6);
        let periods = Array1::zeros(6);
        let min_amps = Array1::zeros(6);
        let weights = Weights::ones(6);

        let result = histogram_one_chunk(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            &weights,
            FrameData::one_frame(6),
            0,
            3000,
            3,
            1000.,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 1, 2, 1, 0, 1]).unwrap();

        assert_eq!(result.hist, expected);
        assert_eq!(result.n, 6)
    }

    /// Test a histogram with filters is correctly constructed.
    #[test]
    fn test_hist_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let amps = Array1::ones(6);
        let periods = Array1::zeros(6);
        let min_amps = Array1::zeros(6);
        // weight bytes are filtering out values 0, 3, 4
        let weights = Weights::from_raw(vec![0b100110]);

        let result = histogram_one_chunk(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            &weights,
            FrameData::one_event_per_frame(6),
            0,
            3000,
            3,
            1000.,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![0, 1, 0, 1, 0, 1]).unwrap();

        assert_eq!(result.hist, expected);
        assert_eq!(result.n, 3)
    }

    /// Test a histogram with multiple periods correctly separates data.
    #[test]
    fn test_hist_multiple_periods() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let amps = Array1::ones(6);
        let periods = Array1::from_vec(vec![0, 0, 1, 1, 0, 1]);
        let min_amps = Array1::zeros(6);
        let weights = Weights::ones(6);

        let result = histogram_one_chunk(
            times,
            specs,
            amps,
            2,
            &periods,
            2,
            &min_amps,
            &weights,
            FrameData::one_event_per_frame(6),
            0,
            3000,
            3,
            1000.,
        );

        // bins are 0-1000, 1000-2000, 2000-3000
        // period 0: events at times 500 (bin 0, spec 0), 600 (bin 0, spec 1), 2500 (bin 2, spec 0)
        // period 1: events at times 1500 (bin 1, spec 0), 2300 (bin 2, spec 0), 2650 (bin 2, spec 1)
        let expected = Array3::<i32>::from_shape_vec(
            (2, 2, 3),
            vec![
                1, 0, 1, // period 0, spec 0
                1, 0, 0, // period 0, spec 1
                0, 1, 1, // period 1, spec 0
                0, 0, 1, // period 1, spec 1
            ],
        )
        .unwrap();

        assert_eq!(result.hist, expected);
        assert_eq!(result.n, 6)
    }

    /// Test a histogram filters out data before the histogram start time.
    #[test]
    fn test_hist_data_before_start() {
        let times = Array1::from_vec(vec![500, 1200, 1800, 2500]);
        let specs = Array1::from_vec(vec![0, 1, 0, 1]);
        let amps = Array1::ones(4);
        let periods = Array1::zeros(4);
        let min_amps = Array1::zeros(4);
        let weights = Weights::ones(4);

        let result = histogram_one_chunk(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            &weights,
            FrameData::one_frame(4),
            1000,
            3000,
            2,
            1000.,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 2), vec![1, 0, 1, 1]).unwrap();

        assert_eq!(result.hist, expected);
        assert_eq!(result.n, 3)
    }

    /// Test a histogram filters out data after the histogram end time.
    #[test]
    fn test_hist_data_after_end() {
        let times = Array1::from_vec(vec![500, 1200, 1800, 3500]);
        let specs = Array1::from_vec(vec![0, 1, 0, 1]);
        let amps = Array1::ones(4);
        let periods = Array1::zeros(4);
        let min_amps = Array1::zeros(4);
        let weights = Weights::ones(4);

        let result = histogram_one_chunk(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            &weights,
            FrameData::one_frame(4),
            0,
            2000,
            2,
            1000.,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 2), vec![1, 1, 0, 1]).unwrap();

        assert_eq!(result.hist, expected);
        assert_eq!(result.n, 3)
    }

    /// Test a histogram with amplitude filters is correctly constructed.
    #[test]
    fn test_hist_amps_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let amps = Array1::from_vec(vec![1., 1., 1., 0.25, 1., 1.75]);
        let periods = Array1::zeros(6);
        let min_amps = Array1::from_vec(vec![0.5, 1.5]);
        let weights = Weights::ones(6);

        let result = histogram_one_chunk(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            &weights,
            FrameData::one_frame(6),
            0,
            3000,
            3,
            1000.,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 1, 1, 0, 0, 1]).unwrap();

        assert_eq!(result.hist, expected);
        assert_eq!(result.n, 4)
    }

    /// Test that two chunks accumulated into one accumulator give the same
    /// result as a single chunk over the same events. This is the property
    /// the per-worker accumulators rest on.
    #[test]
    fn test_accumulator_reuse_across_chunks() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let amps: Array1<f64> = Array1::ones(6);
        let periods = Array1::zeros(6);
        let min_amps = Array1::zeros(6);
        let weights = Weights::ones(6);
        let frame_data = FrameData::new(Array1::from_vec((0..6).collect()), 6);

        // all six events in one chunk
        let single = histogram_one_chunk(
            times.clone(),
            specs.clone(),
            amps.clone(),
            2,
            &periods,
            1,
            &min_amps,
            &weights,
            frame_data.slice(0, 6),
            0,
            3000,
            3,
            1000.,
        );

        // the same six events as two chunks into a single accumulator
        let mut split = Accumulator::zeros(1, 2, 3);
        for (start, end) in [(0, 3), (3, 6)] {
            make_histogram(
                &mut split,
                times.slice(s![start..end]).to_owned(),
                specs.slice(s![start..end]).to_owned(),
                amps.slice(s![start..end]).to_owned(),
                &periods,
                &min_amps,
                &weights,
                frame_data.slice(start, end),
                0,
                3000,
                1000.,
            );
        }

        assert_eq!(split.hist, single.hist);
        assert_eq!(split.n, single.n);

        // the same values the single-chunk tests above pin down
        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 1, 2, 1, 0, 1]).unwrap();
        assert_eq!(split.hist, expected);
        assert_eq!(split.n, 6)
    }

    /// Test that `Accumulator::merge` is order-independent, so the order rayon
    /// happens to fill the accumulator slots in cannot affect the result.
    #[test]
    fn test_merge_order_independent() {
        let first = || Accumulator {
            hist: Array3::from_shape_vec((1, 2, 3), vec![1, 0, 2, 0, 3, 0]).unwrap(),
            n: 6,
        };
        let second = || Accumulator {
            hist: Array3::from_shape_vec((1, 2, 3), vec![0, 4, 0, 5, 0, 6]).unwrap(),
            n: 15,
        };

        let mut first_then_second = first();
        first_then_second.merge(&second());
        let mut second_then_first = second();
        second_then_first.merge(&first());

        assert_eq!(first_then_second.hist, second_then_first.hist);
        assert_eq!(first_then_second.n, second_then_first.n);

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 4, 2, 5, 3, 6]).unwrap();
        assert_eq!(first_then_second.hist, expected);
        assert_eq!(first_then_second.n, 21)
    }

    /// Test that the same file with the same filters gives an identical
    /// histogram and event count at two different chunk sizes, i.e. that the
    /// chunking and accumulator slots do not lose or double-count events.
    #[test]
    fn test_chunk_size_invariance() {
        use crate::interface::Data;

        let calculate = |chunk_size| {
            let mut data =
                Data::new(TEST_FILE.to_string(), 64, chunk_size).expect("failed to open test file");
            data.calculate().expect("failed to calculate histogram");
            data.results
        };

        // the default chunk size, and one small enough to force many chunks
        let default_chunks = calculate(1048576);
        let small_chunks = calculate(65536);

        assert_eq!(small_chunks.hist, default_chunks.hist);
        assert_eq!(small_chunks.n, default_chunks.n);

        // pinned to the fixture, so a change in either shows up as a failure
        // rather than a silent shift
        assert_eq!(default_chunks.n, 64147);
        assert_eq!(default_chunks.hist.sum(), 64147);
    }

    /// Test that `get_period_frames` correctly counts kept frames per period
    /// when all weights are set (no filtering).
    #[test]
    fn test_get_period_frames_no_filter() {
        // 6 frames across 2 periods: [0, 1, 0, 1, 0, 1]
        let periods = Array1::<u32>::from_vec(vec![0, 1, 0, 1, 0, 1]);
        let weights = Weights::ones(6);

        let result = get_period_frames(&periods, 2, &weights);

        assert_eq!(result, vec![3, 3]);
    }

    /// Test that `get_period_frames` correctly ignores frames whose weight
    /// bit is unset (i.e. filtered/vetoed frames).
    #[test]
    fn test_get_period_frames_with_filter() {
        // 6 frames across 2 periods: [0, 1, 0, 1, 0, 1]
        let periods = Array1::<u32>::from_vec(vec![0, 1, 0, 1, 0, 1]);
        // keep frames 0, 1, 2, 4 -> raw bits: 0b010111 (little-endian, LSB = frame 0)
        let weights = Weights::from_raw(vec![0b010111]);

        let result = get_period_frames(&periods, 2, &weights);

        // period 0 frames: indices 0, 2, 4 -> kept: 0, 2, 4 => 3 kept
        // period 1 frames: indices 1, 3, 5 -> kept: 1 only => 1 kept
        assert_eq!(result, vec![3, 1]);
    }

    /// Test that `get_experiment_times` returns the start and end frame
    /// times corresponding to the first and last set bits in the weights,
    /// when all frames are kept.
    #[test]
    fn test_get_experiment_times_no_filter() {
        let frame_start_times = Array1::<usize>::from_vec(vec![100, 200, 300, 400, 500]);
        let weights = Weights::ones(5);

        let (start, end) = get_experiment_times(weights, frame_start_times);

        assert_eq!(start, 100);
        assert_eq!(end, 500);
    }

    /// Test that `get_experiment_times` correctly identifies the start and
    /// end times when only a subset of frames are kept (filtered).
    #[test]
    fn test_get_experiment_times_with_filter() {
        let frame_start_times = Array1::<usize>::from_vec(vec![100, 200, 300, 400, 500]);
        // keep frames 1, 2, 3 only -> raw bits: 0b01110
        let weights = Weights::from_raw(vec![0b01110]);

        let (start, end) = get_experiment_times(weights, frame_start_times);

        assert_eq!(start, 200);
        assert_eq!(end, 400);
    }

    /// Test `__repr__` for a Histogram that has not yet had `calculate` run,
    /// i.e. hist has shape [0, 0, 0].
    #[test]
    fn test_repr_uncalculated() {
        let hist = Histogram::new(500, 2500, 10);
        let repr = hist.__repr__();

        assert!(repr.contains("time range 0.5μs - 2.5μs"));
        assert!(repr.contains("10 bins"));
        assert!(repr.contains("result not calculated"));
    }

    /// Test `__repr__` for a calculated Histogram with a single period uses
    /// the singular "period" (no trailing "s").
    #[test]
    fn test_repr_calculated_single_period() {
        let mut hist = Histogram::new(0, 3000, 3);
        hist.hist = Array3::zeros((1, 4, 3));
        hist.n = 20;
        let repr = hist.__repr__();

        assert!(repr.contains("1 period\n"));
        assert!(!repr.contains("1 periods"));
        assert!(repr.contains("4 detectors"));
        assert!(repr.contains("3 bins"));
        assert!(repr.contains("20 events"));
    }

    /// Test `__repr__` for a calculated Histogram with multiple periods uses
    /// the plural "periods".
    #[test]
    fn test_repr_calculated_multiple_periods() {
        let mut hist = Histogram::new(0, 3000, 3);
        hist.hist = Array3::zeros((2, 4, 3));
        hist.n = 50;
        let repr = hist.__repr__();

        assert!(repr.contains("2 periods"));
    }
}
