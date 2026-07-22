use std::cmp::min;
use std::iter::Iterator;

use anyhow::{Error, Result};
use ndarray::{s, Array1, Array3};
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, Bound};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::data::{FrameData, NexusData};
use crate::filters::{get_weights, Filters, Weights};

type PyHist<'py> = Bound<'py, PyArray3<usize>>;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Histogram {
    pub min_time: f32,
    pub max_time: f32,
    pub n_bins: usize,
    pub hist: Array3<usize>,
    pub n: usize,
}

#[pymethods]
impl Histogram {
    #[new]
    pub fn new(min_time: f32, max_time: f32, n_bins: usize) -> Histogram {
        Histogram {
            min_time,
            max_time,
            n_bins,
            hist: Array3::zeros((0, 0, 0)),
            n: 0,
        }
    }

    fn data<'py>(slf: &Bound<'py, Histogram>) -> PyHist<'py> {
        let py = slf.py();
        slf.borrow().hist.to_pyarray(py)
    }

    fn n_events(&self) -> usize {
        self.n
    }
}

impl Histogram {
    pub fn calculate(&self, data: &NexusData, filters: &Filters) -> Result<Histogram> {
        let periods: Array1<usize> = data.periods.read_1d()?;
        let n_periods = periods.iter().max().unwrap() + 1;

        let start_index: Array1<usize> = data.frames.read_1d()?;
        let frame_data = FrameData::new(start_index.clone(), data.n_events);

        let (time_starts, time_ends) = filters.get_time_filter_times();

        let log_names = filters.get_required_log_names();
        let value_logs = match data.get_sample_logs(log_names) {
            Ok(logs) => logs,
            Err(info) => return Err(Error::msg(format!("Failed to get logs: {info}"))),
        };
        let (log_starts, log_ends) = filters.get_log_filter_times(value_logs);

        let weights = if time_starts.is_empty() && log_starts.is_empty() {
            Weights::ones(data.n_events)
        } else {
            let frame_start_times: Array1<usize> = data.frame_times.read_1d()?;
            let time_weights = if time_starts.is_empty() {
                Weights::ones(data.n_events)
            } else {
                get_weights(
                    time_starts,
                    time_ends,
                    &frame_start_times,
                    &start_index,
                    data.n_events,
                    filters.is_include(),
                )
            };
            // log weights are always include filters
            let log_weights = if log_starts.is_empty() {
                Weights::ones(data.n_events)
            } else {
                get_weights(
                    log_starts,
                    log_ends,
                    &frame_start_times,
                    &start_index,
                    data.n_events,
                    true,
                )
            };
            time_weights & log_weights
        };

        let min_amps = filters.get_amps(data.n_spec)?;

        Ok(calculate_histograms(
            data,
            self.min_time,
            self.max_time,
            self.n_bins,
            n_periods,
            periods,
            min_amps,
            weights,
            frame_data,
        ))
    }
}

/// Calculate histograms and output the result.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn calculate_histograms(
    dataset: &NexusData,
    min_time: f32,
    max_time: f32,
    n_bins: usize,
    n_periods: usize,
    periods: Array1<usize>,
    min_amps: Array1<f64>,
    weights: Weights,
    frame_data: FrameData,
) -> Histogram {
    let width: f64 = (max_time - min_time) as f64 / n_bins as f64;

    // iterate over the data chunks, make histograms for each, then sum histograms at the end
    (0..dataset.n_events)
        .into_par_iter()
        .step_by(dataset.chunk_size)
        .map(|start| {
            let end = min(start + dataset.chunk_size, dataset.n_events);
            let array_slice = s![start..end];
            make_histogram(
                dataset
                    .times
                    .read_slice_1d(array_slice)
                    .expect("Failed to read times."),
                dataset
                    .specs
                    .read_slice_1d(array_slice)
                    .expect("Failed to read specs."),
                dataset
                    .amps
                    .read_slice_1d(array_slice)
                    .expect("Failed to read amplitudes."),
                dataset.n_spec,
                &periods,
                n_periods,
                &min_amps,
                weights.slice(start, end),
                frame_data.slice(start, end),
                min_time,
                max_time,
                n_bins,
                width,
                1e-3,
            )
        })
        // accumulate all chunk histograms together
        .reduce(
            || {
                // rayon's reduce requires to initialise an identity value...
                let mut empty_hist = Histogram::new(min_time, max_time, n_bins);
                empty_hist.hist = Array3::zeros((n_periods, dataset.n_spec, n_bins));
                empty_hist
            },
            |mut acc, r| {
                acc.hist += &r.hist;
                acc.n += &r.n;
                acc
            },
        )
}

/// Make a histogram for a set of data.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn make_histogram(
    times: Array1<u32>,
    specs: Array1<u32>,
    amps: Array1<f64>,
    n_spec: usize,
    periods: &Array1<usize>,
    n_periods: usize,
    min_amps: &Array1<f64>,
    weights: Weights,
    frame_data: FrameData,
    min_time: f32,
    max_time: f32,
    n_bins: usize,
    width: f64,
    conversion: f32,
) -> Histogram {
    let mut result = Histogram::new(min_time, max_time, n_bins);
    result.hist = Array3::zeros((n_periods, n_spec, n_bins));

    let last_frame = frame_data.frame_number.last().unwrap();

    // iterate over the frames in the slice
    for (i, frame) in frame_data.frame_number.iter().enumerate() {
        // get event indices of this frame in the slice
        let frame_start_event = frame_data.start_index[i];
        let frame_end_event = if frame == last_frame {
            frame_data.array_len
        } else {
            frame_data.start_index[i + 1]
        };

        let period = periods[*frame];

        for k in frame_start_event..frame_end_event {
            let t = times[k] as f32 * conversion;
            let amp = amps[k];
            let spec = specs[k] as usize;
            let weight = weights[k];

            if weight && (t >= min_time) && (t <= max_time) && amp > min_amps[spec] {
                let bin = ((t - min_time) / width as f32).floor() as usize;
                result.hist[[period, spec, bin]] += 1;
                result.n += 1
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test Histogram::new creates correct empty histogram.
    #[test]
    fn test_histogram_new() {
        let hist = Histogram::new(0.5, 2.5, 4);
        assert_eq!(hist.min_time, 0.5);
        assert_eq!(hist.max_time, 2.5);
        assert_eq!(hist.n_bins, 4);
        assert_eq!(hist.n, 0);
        assert_eq!(hist.hist.dim(), (0, 0, 0));
    }

    /// Test Histogram::n_events returns correct count.
    #[test]
    fn test_histogram_n_events() {
        let mut hist = Histogram::new(0., 3., 3);
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

        let result = make_histogram(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            weights,
            FrameData::one_frame(6),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<usize>::from_shape_vec((1, 2, 3), vec![1, 1, 2, 1, 0, 1]).unwrap();

        assert_eq!(result.hist, expected)
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

        let result = make_histogram(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            weights,
            FrameData::one_frame(6),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<usize>::from_shape_vec((1, 2, 3), vec![0, 1, 0, 1, 0, 1]).unwrap();

        assert_eq!(result.hist, expected)
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

        let result = make_histogram(
            times,
            specs,
            amps,
            2,
            &periods,
            2,
            &min_amps,
            weights,
            FrameData::one_event_per_frame(6),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        // bins are 0-1000, 1000-2000, 2000-3000
        // period 0: events at times 500 (bin 0, spec 0), 600 (bin 0, spec 1), 2500 (bin 2, spec 0)
        // period 1: events at times 1500 (bin 1, spec 0), 2300 (bin 2, spec 0), 2650 (bin 2, spec 1)
        let expected = Array3::<usize>::from_shape_vec(
            (2, 2, 3),
            vec![
                1, 0, 1, // period 0, spec 0
                1, 0, 0, // period 0, spec 1
                0, 1, 1, // period 1, spec 0
                0, 0, 1, // period 1, spec 1
            ],
        )
        .unwrap();

        assert_eq!(result.hist, expected)
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

        let result = make_histogram(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            weights,
            FrameData::one_frame(4),
            1.,
            3.,
            2,
            1.,
            1e-3,
        );

        let expected = Array3::<usize>::from_shape_vec((1, 2, 2), vec![1, 0, 1, 1]).unwrap();

        assert_eq!(result.hist, expected)
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

        let result = make_histogram(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            weights,
            FrameData::one_frame(4),
            0.,
            2.,
            2,
            1.,
            1e-3,
        );

        let expected = Array3::<usize>::from_shape_vec((1, 2, 2), vec![1, 1, 0, 1]).unwrap();

        assert_eq!(result.hist, expected)
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

        let result = make_histogram(
            times,
            specs,
            amps,
            2,
            &periods,
            1,
            &min_amps,
            weights,
            FrameData::one_frame(6),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<usize>::from_shape_vec((1, 2, 3), vec![1, 1, 1, 0, 0, 1]).unwrap();

        assert_eq!(result.hist, expected)
    }

    /// Test that the conversion value correctly scales time values.
    #[test]
    fn test_hist_conversion_value() {
        let times = Array1::from_vec(vec![400, 800, 2500]);
        let specs = Array1::from_vec(vec![0, 0, 0]);
        let amps = Array1::ones(3);
        let periods = Array1::zeros(3);
        let min_amps = Array1::zeros(3);

        let result = make_histogram(
            times.clone(),
            specs.clone(),
            amps.clone(),
            1,
            &periods,
            1,
            &min_amps,
            Weights::ones(3),
            FrameData::one_frame(3),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<usize>::from_shape_vec((1, 1, 3), vec![2, 0, 1]).unwrap();
        assert_eq!(result.hist, expected);

        // now test with a different conversion factor
        let result2 = make_histogram(
            times,
            specs,
            amps,
            1,
            &periods,
            1,
            &min_amps,
            Weights::ones(3),
            FrameData::one_frame(3),
            0.,
            3.,
            3,
            1.,
            2e-3,
        );

        let expected2 = Array3::<usize>::from_shape_vec((1, 1, 3), vec![1, 1, 0]).unwrap();
        assert_eq!(result2.hist, expected2)
    }
}
