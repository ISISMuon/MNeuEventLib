use std::cmp::min;
use std::iter::Iterator;

use anyhow::{Error, Result};
use ndarray::{s, Array1, Array3};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::data::{FrameData, NexusData};
use crate::filters::{get_weights, Filters, Weights};

#[derive(Clone)]
pub struct Histogram {
    pub min_time: f32,
    pub max_time: f32,
    pub n_bins: usize,
    pub hist: Array3<i32>,
    pub n: usize,
    pub n_frames: Vec<u32>,
    pub n_good_frames: Vec<u32>,
    pub start_time: usize,
    pub end_time: usize,
}

impl Histogram {
    pub fn new(min_time: f32, max_time: f32, n_bins: usize) -> Histogram {
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

    pub fn __repr__(&self) -> String {
        let shape = self.hist.shape();
        let mut string = format!(
            "Histogram with:\n  time range {}, {}",
            self.min_time, self.max_time
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
    min_time: f32,
    max_time: f32,
    n_bins: usize,
    n_periods: usize,
    periods: Array1<u32>,
    min_amps: Array1<f64>,
    weights: &Weights,
    frame_data: FrameData,
) -> Histogram {
    let width: f32 = (max_time - min_time) / n_bins as f32;

    // iterate over the data chunks, make histograms for each, then sum histograms at the end
    (0..dataset.n_events)
        .into_par_iter()
        .step_by(dataset.chunk_size)
        .map(|start| {
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
            make_histogram(
                times,
                specs,
                amps,
                dataset.n_spec,
                &periods,
                n_periods,
                &min_amps,
                weights,
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
    periods: &Array1<u32>,
    n_periods: usize,
    min_amps: &Array1<f64>,
    weights: &Weights,
    frame_data: FrameData,
    min_time: f32,
    max_time: f32,
    n_bins: usize,
    width: f32,
    conversion: f32,
) -> Histogram {
    let mut result = Histogram::new(min_time, max_time, n_bins);
    result.hist = Array3::zeros((n_periods, n_spec, n_bins));

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
        result.n += 1;

        for k in frame_start_event..frame_end_event {
            let t = times[k] as f32 * conversion;
            let amp = amps[k];
            let spec = specs[k] as usize;

            if (t >= min_time) && (t <= max_time) && amp > min_amps[spec] {
                let bin = ((t - min_time) / width).floor() as usize;
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
            &weights,
            FrameData::one_frame(6),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 1, 2, 1, 0, 1]).unwrap();

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
            &weights,
            FrameData::one_event_per_frame(6),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![0, 1, 0, 1, 0, 1]).unwrap();

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
            &weights,
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
            &weights,
            FrameData::one_frame(4),
            1.,
            3.,
            2,
            1.,
            1e-3,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 2), vec![1, 0, 1, 1]).unwrap();

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
            &weights,
            FrameData::one_frame(4),
            0.,
            2.,
            2,
            1.,
            1e-3,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 2), vec![1, 1, 0, 1]).unwrap();

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
            &weights,
            FrameData::one_frame(6),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 1, 1, 0, 0, 1]).unwrap();

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
            &Weights::ones(3),
            FrameData::one_frame(3),
            0.,
            3.,
            3,
            1.,
            1e-3,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 1, 3), vec![2, 0, 1]).unwrap();
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
            &Weights::ones(3),
            FrameData::one_frame(3),
            0.,
            3.,
            3,
            1.,
            2e-3,
        );

        let expected2 = Array3::<i32>::from_shape_vec((1, 1, 3), vec![1, 1, 0]).unwrap();
        assert_eq!(result2.hist, expected2)
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
        let hist = Histogram::new(0.5, 2.5, 10);
        let repr = hist.__repr__();

        assert!(repr.contains("time range 0.5, 2.5"));
        assert!(repr.contains("10 bins"));
        assert!(repr.contains("result not calculated"));
    }

    /// Test `__repr__` for a calculated Histogram with a single period uses
    /// the singular "period" (no trailing "s").
    #[test]
    fn test_repr_calculated_single_period() {
        let mut hist = Histogram::new(0., 3., 3);
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
        let mut hist = Histogram::new(0., 3., 3);
        hist.hist = Array3::zeros((2, 4, 3));
        hist.n = 50;
        let repr = hist.__repr__();

        assert!(repr.contains("2 periods"));
    }
}
