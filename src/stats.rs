use std::cmp::min;
use std::iter::Iterator;

use anyhow::{Error, Result};
use ndarray::{s, Array1, Array3};

use crate::consts::{
    ToMicroseconds, DEFAULT_HIST_MAX_TIME, DEFAULT_HIST_MIN_TIME, DEFAULT_HIST_N_BINS,
};
use crate::data::{FrameData, NexusData};
use crate::filters::{get_log_weights, get_weights, Filters, Weights};

#[derive(Clone)]
pub struct Histogram {
    pub min_time: u32,
    pub max_time: u32,
    pub n_bins: usize,
    pub hist: Array3<i32>,
    pub n: usize,
    pub n_frames: Vec<u32>,
    pub n_good_frames: Vec<u32>,
    pub start_time: u64,
    pub end_time: u64,
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
        if other.n > 0 {
            self.hist += &other.hist;
            self.n += other.n;
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    /// Create an empty histogram with the default binning.
    pub fn new() -> Histogram {
        Histogram::with_bins(
            DEFAULT_HIST_MIN_TIME,
            DEFAULT_HIST_MAX_TIME,
            DEFAULT_HIST_N_BINS,
        )
    }

    /// Create an empty histogram with custom binning.
    pub fn with_bins(min_time: u32, max_time: u32, n_bins: usize) -> Histogram {
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

    /// Empty the calculated result, keeping the histogram settings.
    pub fn reset(&mut self) {
        *self = Histogram::with_bins(self.min_time, self.max_time, self.n_bins);
    }
    /// Estimate the end of the run, as the end of the histogram window after the last frame
    /// starts (only frame starts are stored, not ends).
    pub fn run_end(&self, frame_start_times: &Array1<u64>) -> u64 {
        frame_start_times.last().unwrap() + self.max_time as u64
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

        let filters_exist = !time_starts.is_empty() || !filters.sample_log_filters.is_empty();

        let frame_start_times: Array1<u64> = data.frame_times.read_1d()?;

        let run_end = self.run_end(&frame_start_times);

        let weights = if filters_exist {
            let time_weights = if time_starts.is_empty() {
                Weights::ones(data.n_frames)
            } else {
                get_weights(
                    time_starts,
                    time_ends,
                    &frame_start_times,
                    run_end,
                    filters.is_include(),
                    "time filter",
                )
            };
            // log weights are always include filters
            let log_weights = if filters.sample_log_filters.is_empty() {
                Weights::ones(data.n_frames)
            } else {
                // get data for all log filters that have been filtered
                let log_names = filters.get_required_log_names();
                let value_logs = match data.get_sample_logs(log_names) {
                    Ok(logs) => logs,
                    Err(info) => return Err(Error::msg(format!("Failed to get logs: {info}"))),
                };
                let log_filter_times = filters.get_log_filter_times(value_logs)?;
                get_log_weights(log_filter_times, &frame_start_times, run_end)
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
pub fn get_experiment_times(weights: Weights, frame_start_times: Array1<u64>) -> (u64, u64) {
    match (weights.get_first_one(), weights.get_last_one()) {
        (Some(first), Some(last)) => (frame_start_times[first], frame_start_times[last]),
        // return 0 time when all frames have been filtered out rather than crashing
        _ => (0, 0),
    }
}

/// In-memory chunk of raw event arrays passed through the pipelined reader channel.
struct ChunkData {
    /// Event Time of Flight (in nanoseconds).
    times: Array1<u32>,
    /// Detector spectrum IDs.
    specs: Array1<u32>,
    /// Pulse height amplitude values.
    amps: Array1<f64>,
    /// Global start index of this chunk within the dataset.
    start_event_index: usize,
    /// Number of events in this chunk.
    chunk_length: usize,
    /// Lowest frame index containing events belonging to this chunk.
    lower_frame_index: usize,
    /// Highest frame index containing events belonging to this chunk.
    upper_frame_index: usize,
}

/// Calculate histograms and output the result using a pipelined reader and lock-free parallel workers.
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
    // Precompute reciprocal of bin width to replace inner loop floating-point divisions with fast multiplications
    let bin_width: f32 = (max_time - min_time) as f32 / n_bins as f32;
    let inv_bin_width: f32 = 1.0f32 / bin_width;
    let has_amp_filter = min_amps.iter().any(|&threshold| threshold > 0.0);

    let frame_data_ref = &frame_data;
    let frame_starts = &frame_data.start_index;
    let periods_ref = &periods;
    let min_amps_ref = &min_amps;
    let n_spec = dataset.n_spec;

    let total_chunks = if dataset.chunk_size > 0 {
        dataset.n_events.div_ceil(dataset.chunk_size)
    } else {
        1
    };
    let num_workers = rayon::current_num_threads().clamp(1, total_chunks.max(1));

    // Bounded channel to decouple sequential HDF5 I/O from parallel CPU histogramming
    let (chunk_sender, chunk_receiver) = std::sync::mpsc::sync_channel::<ChunkData>(4);
    let shared_receiver = std::sync::Arc::new(std::sync::Mutex::new(chunk_receiver));

    let final_accumulator = std::thread::scope(|scope| {
        // Dedicated sequential HDF5 reader thread: eliminates HDF5 C global mutex contention
        scope.spawn(move || {
            for start_index in (0..dataset.n_events).step_by(dataset.chunk_size) {
                let end_index = min(start_index + dataset.chunk_size, dataset.n_events);
                let array_slice = s![start_index..end_index];
                let amps: Array1<f64> = dataset
                    .amps
                    .read_slice_1d(array_slice)
                    .expect("Failed to read amplitudes from dataset.");
                let times: Array1<u32> = dataset
                    .times
                    .read_slice_1d(array_slice)
                    .expect("Failed to read times from dataset.");
                let specs: Array1<u32> = dataset
                    .specs
                    .read_slice_1d(array_slice)
                    .expect("Failed to read specs from dataset.");

                let (lower_frame_index, upper_frame_index) =
                    frame_data_ref.frame_range(start_index, end_index);

                let chunk = ChunkData {
                    times,
                    specs,
                    amps,
                    start_event_index: start_index,
                    chunk_length: end_index - start_index,
                    lower_frame_index,
                    upper_frame_index,
                };

                if chunk_sender.send(chunk).is_err() {
                    break;
                }
            }
        });

        // Worker threads pull pre-read chunk buffers asynchronously and accumulate thread-locally
        let mut worker_handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let thread_receiver = std::sync::Arc::clone(&shared_receiver);
            let handle = scope.spawn(move || {
                let mut local_accumulator = Accumulator::zeros(n_periods, n_spec, n_bins);
                loop {
                    let next_chunk = {
                        let locked_receiver = thread_receiver.lock().unwrap();
                        locked_receiver.recv().ok()
                    };
                    match next_chunk {
                        Some(chunk) => {
                            make_histogram(
                                &mut local_accumulator,
                                &chunk,
                                periods_ref,
                                min_amps_ref,
                                weights,
                                frame_starts,
                                min_time,
                                max_time,
                                inv_bin_width,
                                n_bins,
                                n_spec,
                                has_amp_filter,
                            );
                        }
                        None => break,
                    }
                }
                local_accumulator
            });
            worker_handles.push(handle);
        }

        // Merge thread-local accumulators at completion
        let mut combined_accumulator = Accumulator::zeros(n_periods, n_spec, n_bins);
        for handle in worker_handles {
            let worker_acc = handle.join().unwrap();
            combined_accumulator.merge(&worker_acc);
        }
        combined_accumulator
    });

    let mut histogram = Histogram::with_bins(min_time, max_time, n_bins);
    histogram.hist = final_accumulator.hist;
    histogram.n = final_accumulator.n;
    histogram
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn make_histogram(
    acc: &mut Accumulator,
    chunk: &ChunkData,
    periods: &Array1<u32>,
    min_amps: &Array1<f64>,
    weights: &Weights,
    frame_starts: &Array1<usize>,
    min_time: u32,
    max_time: u32,
    inv_bin_width: f32,
    n_bins: usize,
    n_spec: usize,
    has_amp_filter: bool,
) {
    // Extract raw contiguous slices to enable linear 1D indexing and auto-vectorization
    let times_slice = chunk.times.as_slice().unwrap();
    let specs_slice = chunk.specs.as_slice().unwrap();
    let amps_slice = chunk.amps.as_slice().unwrap();
    let periods_slice = periods.as_slice().unwrap();
    let min_amps_slice = min_amps.as_slice().unwrap();
    let frame_starts_slice = frame_starts.as_slice().unwrap();
    let hist_slice = acc.hist.as_slice_mut().unwrap();

    let lower_frame = chunk.lower_frame_index;
    let upper_frame = chunk.upper_frame_index;
    let chunk_start = chunk.start_event_index;
    let chunk_len = chunk.chunk_length;

    let stride_period = n_spec * n_bins;
    let mut total_events_accepted = 0;

    if has_amp_filter {
        // Fast path for active amplitude filter
        for frame in lower_frame..=upper_frame {
            if !weights[frame] {
                continue;
            }

            let frame_start_event = if frame == lower_frame {
                0
            } else {
                frame_starts_slice[frame] - chunk_start
            };
            let frame_end_event = if frame == upper_frame {
                chunk_len
            } else {
                frame_starts_slice[frame + 1] - chunk_start
            };

            let period_idx = periods_slice[frame] as usize;
            let period_offset = period_idx * stride_period;

            for event_idx in frame_start_event..frame_end_event {
                let event_time = times_slice[event_idx];
                let pulse_amp = amps_slice[event_idx];
                let spec_idx = specs_slice[event_idx] as usize;

                if event_time >= min_time
                    && event_time < max_time
                    && pulse_amp > min_amps_slice[spec_idx]
                {
                    let bin_idx = ((event_time - min_time) as f32 * inv_bin_width) as usize;
                    let bin_idx = min(bin_idx, n_bins - 1);
                    hist_slice[period_offset + spec_idx * n_bins + bin_idx] += 1;
                    total_events_accepted += 1;
                }
            }
        }
    } else {
        // Fast path when no detector has an amplitude threshold (skips per-detector threshold lookup)
        for frame in lower_frame..=upper_frame {
            if !weights[frame] {
                continue;
            }

            let frame_start_event = if frame == lower_frame {
                0
            } else {
                frame_starts_slice[frame] - chunk_start
            };
            let frame_end_event = if frame == upper_frame {
                chunk_len
            } else {
                frame_starts_slice[frame + 1] - chunk_start
            };

            let period_idx = periods_slice[frame] as usize;
            let period_offset = period_idx * stride_period;

            for event_idx in frame_start_event..frame_end_event {
                let event_time = times_slice[event_idx];
                let pulse_amp = amps_slice[event_idx];
                let spec_idx = specs_slice[event_idx] as usize;

                if event_time >= min_time && event_time < max_time {
                    let bin_idx = ((event_time - min_time) as f32 * inv_bin_width) as usize;
                    let bin_idx = min(bin_idx, n_bins - 1);
                    hist_slice[period_offset + spec_idx * n_bins + bin_idx] += 1;
                    total_events_accepted += 1;
                }
            }
        }
    }

    acc.n += total_events_accepted;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a histogram over one chunk of data.
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
        let n_events = times.len();
        let inv_width = 1.0f32 / width;
        let has_amp_filter = min_amps.iter().any(|&a| a > 0.0);
        let (lower_frame, upper_frame) = if frame_data.start_index.is_empty() {
            (0, 0)
        } else {
            frame_data.frame_range(0, n_events)
        };
        let chunk = ChunkData {
            times,
            specs,
            amps,
            start_event_index: 0,
            chunk_length: n_events,
            lower_frame_index: lower_frame,
            upper_frame_index: upper_frame,
        };
        make_histogram(
            &mut acc,
            &chunk,
            periods,
            min_amps,
            weights,
            &frame_data.start_index,
            min_time,
            max_time,
            inv_width,
            n_bins,
            n_spec,
            has_amp_filter,
        );
        acc
    }

    /// Test Histogram::with_bins creates correct empty histogram.
    #[test]
    fn test_histogram_with_bins() {
        let hist = Histogram::with_bins(500, 2500, 4);
        assert_eq!(hist.min_time, 500);
        assert_eq!(hist.max_time, 2500);
        assert_eq!(hist.n_bins, 4);
        assert_eq!(hist.n, 0);
        assert_eq!(hist.hist.dim(), (0, 0, 0));
    }

    /// Test Histogram::new uses the default binning.
    #[test]
    fn test_histogram_new() {
        let hist = Histogram::new();
        assert_eq!(hist.min_time, DEFAULT_HIST_MIN_TIME);
        assert_eq!(hist.max_time, DEFAULT_HIST_MAX_TIME);
        assert_eq!(hist.n_bins, DEFAULT_HIST_N_BINS);
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
    /// result as a single chunk over the same events
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
        let inv_width = 1.0f32 / 1000.;
        let has_amp_filter = min_amps.iter().any(|&a| a > 0.0);
        for (start, end) in [(0, 3), (3, 6)] {
            let (lower_frame, upper_frame) = frame_data.frame_range(start, end);
            let t_slice = times.slice(s![start..end]).to_owned();
            let s_slice = specs.slice(s![start..end]).to_owned();
            let a_slice = amps.slice(s![start..end]).to_owned();
            let chunk = ChunkData {
                times: t_slice,
                specs: s_slice,
                amps: a_slice,
                start_event_index: start,
                chunk_length: end - start,
                lower_frame_index: lower_frame,
                upper_frame_index: upper_frame,
            };
            make_histogram(
                &mut split,
                &chunk,
                &periods,
                &min_amps,
                &weights,
                &frame_data.start_index,
                0,
                3000,
                inv_width,
                3,
                2,
                has_amp_filter,
            );
        }

        assert_eq!(split.hist, single.hist);
        assert_eq!(split.n, single.n);

        // the same values the single-chunk tests above pin down
        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 1, 2, 1, 0, 1]).unwrap();
        assert_eq!(split.hist, expected);
        assert_eq!(split.n, 6)
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
        let frame_start_times = Array1::<u64>::from_vec(vec![100, 200, 300, 400, 500]);
        let weights = Weights::ones(5);

        let (start, end) = get_experiment_times(weights, frame_start_times);

        assert_eq!(start, 100);
        assert_eq!(end, 500);
    }

    /// Test that `get_experiment_times` returns zeroes rather than panicking
    /// when the filters keep no frames at all.
    #[test]
    fn test_get_experiment_times_no_frames() {
        let frame_start_times = Array1::<u64>::from_vec(vec![100, 200, 300, 400, 500]);
        let weights = Weights::zeros(5);

        let (start, end) = get_experiment_times(weights, frame_start_times);

        assert_eq!(start, 0);
        assert_eq!(end, 0);
    }

    /// Test that `get_experiment_times` correctly identifies the start and
    /// end times when only a subset of frames are kept (filtered).
    #[test]
    fn test_get_experiment_times_with_filter() {
        let frame_start_times = Array1::<u64>::from_vec(vec![100, 200, 300, 400, 500]);
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
        let hist = Histogram::with_bins(500, 2500, 10);
        let repr = hist.__repr__();

        assert!(repr.contains("time range 0.5μs - 2.5μs"));
        assert!(repr.contains("10 bins"));
        assert!(repr.contains("result not calculated"));
    }

    /// Test `__repr__` for a calculated Histogram with a single period uses
    /// the singular "period" (no trailing "s").
    #[test]
    fn test_repr_calculated_single_period() {
        let mut hist = Histogram::with_bins(0, 3000, 3);
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
        let mut hist = Histogram::with_bins(0, 3000, 3);
        hist.hist = Array3::zeros((2, 4, 3));
        hist.n = 50;
        let repr = hist.__repr__();

        assert!(repr.contains("2 periods"));
    }
}
