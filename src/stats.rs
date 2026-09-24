use std::cmp::min;
use std::iter::Iterator;

use anyhow::{Error, Result};
use ndarray::{s, Array1, Array3};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::consts::ToMicroseconds;
use crate::data::{FrameData, NexusData};
use crate::filters::{get_weights, Filters, Weights};
use crate::gpu::{DevicePreference, GpuContext, GpuHistogrammer};

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

    /// Calculate histograms using the default device preference (`DevicePreference::Auto`).
    pub fn calculate(&self, data: &NexusData, filters: &Filters) -> Result<Histogram> {
        self.calculate_with_device(data, filters, DevicePreference::Auto)
    }

    /// Calculate histograms using a specific device preference (`Auto`, `Cpu`, `Gpu`, or `Hybrid`).
    pub fn calculate_with_device(
        &self,
        data: &NexusData,
        filters: &Filters,
        device: DevicePreference,
    ) -> Result<Histogram> {
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
            device,
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
pub fn get_experiment_times(weights: Weights, frame_start_times: Array1<usize>) -> (usize, usize) {
    (
        frame_start_times[weights.get_first_one().unwrap()],
        frame_start_times[weights.get_last_one().unwrap()],
    )
}

/// Calculate histograms from event datasets according to the requested device preference.
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
    device: DevicePreference,
) -> Histogram {
    let effective_device = match device {
        DevicePreference::Auto => {
            if GpuContext::get().is_some() {
                DevicePreference::Gpu
            } else {
                DevicePreference::Cpu
            }
        }
        other => other,
    };

    match effective_device {
        DevicePreference::Cpu => calculate_histograms_cpu(
            dataset, min_time, max_time, n_bins, n_periods, &periods, &min_amps, weights, &frame_data,
        ),
        DevicePreference::Gpu => {
            if let Some(ctx) = GpuContext::get() {
                match calculate_histograms_gpu(
                    dataset, min_time, max_time, n_bins, n_periods, &periods, &min_amps, weights, &frame_data, ctx,
                ) {
                    Ok(hist) => hist,
                    Err(err) => {
                        eprintln!("GPU calculation failed ({err:?}), falling back to CPU");
                        calculate_histograms_cpu(
                            dataset, min_time, max_time, n_bins, n_periods, &periods, &min_amps, weights, &frame_data,
                        )
                    }
                }
            } else {
                calculate_histograms_cpu(
                    dataset, min_time, max_time, n_bins, n_periods, &periods, &min_amps, weights, &frame_data,
                )
            }
        }
        DevicePreference::Hybrid => {
            if let Some(ctx) = GpuContext::get() {
                match calculate_histograms_hybrid(
                    dataset, min_time, max_time, n_bins, n_periods, &periods, &min_amps, weights, &frame_data, ctx,
                ) {
                    Ok(hist) => hist,
                    Err(err) => {
                        eprintln!("Hybrid calculation failed ({err:?}), falling back to CPU");
                        calculate_histograms_cpu(
                            dataset, min_time, max_time, n_bins, n_periods, &periods, &min_amps, weights, &frame_data,
                        )
                    }
                }
            } else {
                calculate_histograms_cpu(
                    dataset, min_time, max_time, n_bins, n_periods, &periods, &min_amps, weights, &frame_data,
                )
            }
        }
        DevicePreference::Auto => unreachable!(),
    }
}

/// Calculate histograms on CPU using Rayon multi-threading.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn calculate_histograms_cpu(
    dataset: &NexusData,
    min_time: u32,
    max_time: u32,
    n_bins: usize,
    n_periods: usize,
    periods: &Array1<u32>,
    min_amps: &Array1<f64>,
    weights: &Weights,
    frame_data: &FrameData,
) -> Histogram {
    let width: f32 = (max_time - min_time) as f32 / n_bins as f32;
    let inv_width: f32 = 1.0 / width;

    (0..dataset.n_events)
        .into_par_iter()
        .step_by(dataset.chunk_size)
        .fold(
            || {
                let mut acc = Histogram::new(min_time, max_time, n_bins);
                acc.hist = Array3::zeros((n_periods, dataset.n_spec, n_bins));
                acc
            },
            |mut acc, start| {
                let end = min(start + dataset.chunk_size, dataset.n_events);
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
                    &mut acc,
                    times,
                    specs,
                    amps,
                    periods,
                    n_periods,
                    min_amps,
                    weights,
                    frame_data.slice(start, end),
                    min_time,
                    max_time,
                    inv_width,
                );
                acc
            },
        )
        .reduce(
            || {
                let mut empty_hist = Histogram::new(min_time, max_time, n_bins);
                empty_hist.hist = Array3::zeros((n_periods, dataset.n_spec, n_bins));
                empty_hist
            },
            |mut acc, r| {
                acc.hist += &r.hist;
                acc.n += r.n;
                acc
            },
        )
}

/// Calculate histograms on GPU with pipelined HDF5 chunk reads.
#[allow(clippy::too_many_arguments)]
fn calculate_histograms_gpu(
    dataset: &NexusData,
    min_time: u32,
    max_time: u32,
    n_bins: usize,
    n_periods: usize,
    periods: &Array1<u32>,
    min_amps: &Array1<f64>,
    weights: &Weights,
    frame_data: &FrameData,
    ctx: &'static GpuContext,
) -> Result<Histogram> {
    let width: f32 = (max_time - min_time) as f32 / n_bins as f32;
    let inv_width: f32 = 1.0 / width;
    let min_amps_f32: Vec<f32> = min_amps.iter().map(|&a| a as f32).collect();

    let gpu_hist = GpuHistogrammer::new(
        ctx,
        n_periods,
        dataset.n_spec,
        n_bins,
        &min_amps_f32,
        dataset.chunk_size,
    )?;

    struct PreparedGpuChunk {
        times: Array1<u32>,
        specs: Array1<u32>,
        amps: Array1<f32>,
        periods: Vec<u32>,
    }

    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<PreparedGpuChunk>>(2);

    std::thread::scope(|s| -> Result<()> {
        let producer = s.spawn(move || -> Result<()> {
            for start in (0..dataset.n_events).step_by(dataset.chunk_size) {
                let end = min(start + dataset.chunk_size, dataset.n_events);
                let array_slice = s![start..end];
                let amps: Array1<f32> = dataset.amps.read_slice_1d(array_slice)?;
                let times: Array1<u32> = dataset.times.read_slice_1d(array_slice)?;
                let specs: Array1<u32> = dataset.specs.read_slice_1d(array_slice)?;

                let chunk_frame_data = frame_data.slice(start, end);
                let chunk_len = chunk_frame_data.array_len;
                let mut event_periods = vec![u32::MAX; chunk_len];

                if let Some(&last_frame) = chunk_frame_data.frame_number.last() {
                    for (i, &frame) in chunk_frame_data.frame_number.iter().enumerate() {
                        if !weights[frame] {
                            continue;
                        }
                        let frame_start = chunk_frame_data.start_index[i];
                        let frame_end = if frame == last_frame {
                            chunk_frame_data.array_len
                        } else {
                            chunk_frame_data.start_index[i + 1]
                        };
                        let p = periods[frame];
                        event_periods[frame_start..frame_end].fill(p);
                    }
                }

                if tx
                    .send(Ok(PreparedGpuChunk {
                        times,
                        specs,
                        amps,
                        periods: event_periods,
                    }))
                    .is_err()
                {
                    break;
                }
            }
            Ok(())
        });

        for chunk_res in rx {
            let chunk = chunk_res?;
            gpu_hist.dispatch_chunk(
                min_time,
                max_time,
                inv_width,
                chunk.times.as_slice().unwrap(),
                chunk.specs.as_slice().unwrap(),
                chunk.amps.as_slice().unwrap(),
                &chunk.periods,
            )?;
        }

        producer.join().unwrap()
    })?;


    let (hist, n) = gpu_hist.readback(n_periods)?;
    let mut result = Histogram::new(min_time, max_time, n_bins);
    result.hist = hist;
    result.n = n;
    Ok(result)
}

/// Calculate histograms using hybrid CPU + GPU co-processing.
#[allow(clippy::too_many_arguments)]
fn calculate_histograms_hybrid(
    dataset: &NexusData,
    min_time: u32,
    max_time: u32,
    n_bins: usize,
    n_periods: usize,
    periods: &Array1<u32>,
    min_amps: &Array1<f64>,
    weights: &Weights,
    frame_data: &FrameData,
    ctx: &'static GpuContext,
) -> Result<Histogram> {
    let width: f32 = (max_time - min_time) as f32 / n_bins as f32;
    let inv_width: f32 = 1.0 / width;
    let min_amps_f32: Vec<f32> = min_amps.iter().map(|&a| a as f32).collect();

    let gpu_hist = GpuHistogrammer::new(
        ctx,
        n_periods,
        dataset.n_spec,
        n_bins,
        &min_amps_f32,
        dataset.chunk_size,
    )?;

    let mut cpu_acc = Histogram::new(min_time, max_time, n_bins);
    cpu_acc.hist = Array3::zeros((n_periods, dataset.n_spec, n_bins));

    let mut gpu_event_periods = vec![u32::MAX; dataset.chunk_size];

    for start in (0..dataset.n_events).step_by(dataset.chunk_size) {
        let end = min(start + dataset.chunk_size, dataset.n_events);
        let chunk_len = end - start;
        let array_slice = s![start..end];

        let amps: Array1<f32> = dataset.amps.read_slice_1d(array_slice)?;
        let times: Array1<u32> = dataset.times.read_slice_1d(array_slice)?;
        let specs: Array1<u32> = dataset.specs.read_slice_1d(array_slice)?;

        let split = chunk_len / 2;

        // GPU processes second half [split..chunk_len]
        let gpu_start = start + split;
        let gpu_end = end;
        let gpu_len = gpu_end - gpu_start;

        if gpu_len > 0 {
            let gpu_frame_data = frame_data.slice(gpu_start, gpu_end);
            gpu_event_periods.clear();
            gpu_event_periods.resize(gpu_len, u32::MAX);

            if let Some(&last_frame) = gpu_frame_data.frame_number.last() {
                for (i, &frame) in gpu_frame_data.frame_number.iter().enumerate() {
                    if !weights[frame] {
                        continue;
                    }
                    let frame_start = gpu_frame_data.start_index[i];
                    let frame_end = if frame == last_frame {
                        gpu_frame_data.array_len
                    } else {
                        gpu_frame_data.start_index[i + 1]
                    };
                    let p = periods[frame];
                    gpu_event_periods[frame_start..frame_end].fill(p);
                }
            }

            let amps_slice = amps.as_slice().unwrap();
            let times_slice = times.as_slice().unwrap();
            let specs_slice = specs.as_slice().unwrap();

            gpu_hist.dispatch_chunk(
                min_time,
                max_time,
                inv_width,
                &times_slice[split..chunk_len],
                &specs_slice[split..chunk_len],
                &amps_slice[split..chunk_len],
                &gpu_event_periods,
            )?;
        }

        // CPU processes first half [0..split] concurrently while GPU is executing
        if split > 0 {
            let cpu_frame_data = frame_data.slice(start, start + split);
            let times_slice = times.as_slice().unwrap();
            let specs_slice = specs.as_slice().unwrap();
            let amps_slice = amps.as_slice().unwrap();

            bin_events_slice(
                &mut cpu_acc,
                &times_slice[0..split],
                &specs_slice[0..split],
                &amps_slice[0..split],
                periods,
                &min_amps_f32,
                weights,
                &cpu_frame_data,
                min_time,
                max_time,
                inv_width,
            );
        }
    }

    let (gpu_hist_arr, gpu_n) = gpu_hist.readback(n_periods)?;
    cpu_acc.hist += &gpu_hist_arr;
    cpu_acc.n += gpu_n;

    Ok(cpu_acc)
}

/// Accumulate events from contiguous slices of chunk arrays into a histogram on CPU.
#[inline(always)]
fn bin_events_slice<T: Copy + PartialOrd>(
    result: &mut Histogram,
    times: &[u32],
    specs: &[u32],
    amps: &[T],
    periods: &Array1<u32>,
    min_amps: &[T],
    weights: &Weights,
    frame_data: &FrameData,
    min_time: u32,
    max_time: u32,
    inv_width: f32,
) {
    if frame_data.frame_number.is_empty() {
        return;
    }
    let last_frame = *frame_data.frame_number.last().unwrap();

    for (i, &frame) in frame_data.frame_number.iter().enumerate() {
        if !weights[frame] {
            continue;
        }

        let frame_start_event = frame_data.start_index[i];
        let frame_end_event = if frame == last_frame {
            frame_data.array_len
        } else {
            frame_data.start_index[i + 1]
        };

        let period = periods[frame] as usize;

        for k in frame_start_event..frame_end_event {
            let t = times[k];
            let amp = amps[k];
            let spec = specs[k] as usize;

            if (t >= min_time) && (t < max_time) && amp > min_amps[spec] {
                let bin = ((t - min_time) as f32 * inv_width) as usize;
                result.hist[[period, spec, bin]] += 1;
                result.n += 1;
            }
        }
    }
}

/// Bin a set of data directly into the given accumulating histogram.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn make_histogram(
    result: &mut Histogram,
    times: Array1<u32>,
    specs: Array1<u32>,
    amps: Array1<f64>,
    periods: &Array1<u32>,
    n_periods: usize,
    min_amps: &Array1<f64>,
    weights: &Weights,
    frame_data: FrameData,
    min_time: u32,
    max_time: u32,
    inv_width: f32,
) {
    let _ = n_periods;
    bin_events_slice(
        result,
        times.as_slice().unwrap(),
        specs.as_slice().unwrap(),
        amps.as_slice().unwrap(),
        periods,
        min_amps.as_slice().unwrap(),
        weights,
        &frame_data,
        min_time,
        max_time,
        inv_width,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: allocate a histogram, bin the data into it, and return it.
    #[allow(clippy::too_many_arguments)]
    fn build_histogram(
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
        inv_width: f32,
    ) -> Histogram {
        let mut result = Histogram::new(min_time, max_time, n_bins);
        result.hist = Array3::zeros((n_periods, n_spec, n_bins));
        make_histogram(
            &mut result,
            times,
            specs,
            amps,
            periods,
            n_periods,
            min_amps,
            weights,
            frame_data,
            min_time,
            max_time,
            inv_width,
        );
        result
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

    /// Test a histogram with no filters is correctly constructed.
    #[test]
    fn test_hist_no_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let amps = Array1::ones(6);
        let periods = Array1::zeros(6);
        let min_amps = Array1::zeros(6);
        let weights = Weights::ones(6);

        let result = build_histogram(
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
            0.001,
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

        let result = build_histogram(
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
            0.001,
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

        let result = build_histogram(
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
            0.001,
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

        let result = build_histogram(
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
            0.001,
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

        let result = build_histogram(
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
            0.001,
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

        let result = build_histogram(
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
            0.001,
        );

        let expected = Array3::<i32>::from_shape_vec((1, 2, 3), vec![1, 1, 1, 0, 0, 1]).unwrap();

        assert_eq!(result.hist, expected);
        assert_eq!(result.n, 4)
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

    #[test]
    fn test_device_preference_parse() {
        assert_eq!(DevicePreference::from_str("auto").unwrap(), DevicePreference::Auto);
        assert_eq!(DevicePreference::from_str("CPU").unwrap(), DevicePreference::Cpu);
        assert_eq!(DevicePreference::from_str("gpu").unwrap(), DevicePreference::Gpu);
        assert_eq!(DevicePreference::from_str("Hybrid ").unwrap(), DevicePreference::Hybrid);
        assert!(DevicePreference::from_str("invalid").is_err());
    }

    #[test]
    fn test_calculate_device_parity() {
        let _guard = crate::test_utils::lock_hdf5_test();
        const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";
        let dataset = NexusData::new(TEST_FILE.to_string(), 64, 1048576).unwrap();
        let filters = Filters::new();
        let base_hist = Histogram::new(0, 32768, 2048);

        let cpu_hist = base_hist
            .calculate_with_device(&dataset, &filters, DevicePreference::Cpu)
            .unwrap();
        assert!(cpu_hist.n > 0);

        if GpuContext::get().is_some() {
            let gpu_hist = base_hist
                .calculate_with_device(&dataset, &filters, DevicePreference::Gpu)
                .unwrap();
            assert_eq!(cpu_hist.n, gpu_hist.n, "Event counts must match between CPU and GPU");
            assert_eq!(cpu_hist.hist, gpu_hist.hist, "Histogram arrays must match between CPU and GPU");

            let hybrid_hist = base_hist
                .calculate_with_device(&dataset, &filters, DevicePreference::Hybrid)
                .unwrap();
            assert_eq!(cpu_hist.n, hybrid_hist.n, "Event counts must match between CPU and Hybrid");
            assert_eq!(cpu_hist.hist, hybrid_hist.hist, "Histogram arrays must match between CPU and Hybrid");
        }
    }
}
