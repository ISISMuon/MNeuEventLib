use std::cmp::min;
use std::iter::Iterator;
use std::time::{Duration, Instant};

use ndarray::{s, Array1, Array3, ArrayView1};
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, Bound, PyResult};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::data::NexusData;
use crate::filters::Weights;

type PyHist<'py> = Bound<'py, PyArray3<usize>>;

#[pyclass(frozen)]
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
    fn new(min_time: f32, max_time: f32, n_bins: usize) -> Histogram {
        Histogram {
            min_time,
            max_time,
            n_bins,
            hist: Array3::zeros((0, 0, 0)),
            n: 0,
        }
    }

    fn data<'py>(slf: &Bound<'py, Histogram>) -> PyResult<PyHist<'py>> {
        let py = slf.py();
        Ok(slf.borrow().hist.to_pyarray(py))
    }

    fn n_events(&self) -> PyResult<usize> {
        Ok(self.n)
    }

    // todo: change n_filters to a proper Filters object
    fn calculate(&self, data: NexusData) -> PyResult<(Histogram, u128)> {
        // todo: add periods
        let periods: Array1<u32> = Array1::zeros(data.n_events);
        let n_periods: usize = *periods.iter().max().unwrap() as usize + 1;

        let weights = Weights::ones(data.n_events);

        let (result, time) = calculate_histograms(
            data,
            self.min_time,
            self.max_time,
            self.n_bins,
            n_periods,
            periods,
            weights,
        );
        Ok((result, time.as_millis()))
    }
}

/// Calculate histograms and output the result and time taken.
#[inline(always)]
pub fn calculate_histograms(
    dataset: NexusData,
    min_time: f32,
    max_time: f32,
    n_bins: usize,
    n_periods: usize,
    periods: Array1<u32>,
    weights: Weights,
) -> (Histogram, Duration) {
    let time = Instant::now();

    let width: f64 = (max_time - min_time) as f64 / n_bins as f64;

    // iterate over the data chunks, make histograms for each, then sum histograms at the end
    let results: Histogram = (0..dataset.n_events)
        .into_par_iter()
        .step_by(dataset.chunk_size)
        .map(|start| {
            let end = start + min(dataset.chunk_size, dataset.n_events - start);
            let array_slice = s![start..end];
            unsafe {
                make_histogram(
                    dataset
                        .times
                        .read_slice_1d(array_slice)
                        .expect("Failed to read times."),
                    dataset
                        .specs
                        .read_slice_1d(array_slice)
                        .expect("Failed to read specs."),
                    dataset.n_spec,
                    &periods.slice(array_slice),
                    n_periods,
                    weights.slice(start, end),
                    min_time,
                    max_time,
                    n_bins,
                    width,
                    1e-3,
                )
            }
        })
        .reduce(
            || {
                // rayon's reduce requires to initialise a value...
                let mut h = Histogram::new(min_time, max_time, n_bins);
                h.hist = Array3::zeros((n_periods, dataset.n_spec, n_bins));
                h
            },
            |mut acc, r| {
                acc.hist += &r.hist;
                acc.n += &r.n;
                acc
            },
        );

    (results, time.elapsed())
}

/// Make a histogram for a set of data.
/// This function is unsafe because we do array indexing without bounds checks!
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn make_histogram(
    times: Array1<u32>,
    specs: Array1<u32>,
    n_spec: usize,
    periods: &ArrayView1<u32>,
    n_periods: usize,
    weights: Weights,
    min_time: f32,
    max_time: f32,
    n_bins: usize,
    width: f64,
    conversion: f32,
) -> Histogram {
    let mut result = Histogram::new(min_time, max_time, n_bins);
    result.hist = Array3::zeros((n_periods, n_spec, n_bins));

    for (k, time) in times.into_iter().enumerate() {
        let t = time as f32 * conversion;
        let w_k = weights[k];

        if w_k && (t >= min_time) && (t <= max_time) {
            let bin = ((t - min_time) / width as f32).floor() as usize;
            result.hist[[*periods.uget(k) as usize, *specs.uget(k) as usize, bin]] += 1;
            result.n += 1
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test a histogram with no filters is correctly constructed.
    #[test]
    fn test_hist_no_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let periods = Array1::zeros(6);
        let weights = Weights::ones(6);

        unsafe {
            let result = make_histogram(
                times,
                specs,
                2,
                &periods.slice(s![0..=5]),
                1,
                weights,
                0.,
                3.,
                3,
                1.,
                1e-3,
            );

            let expected =
                Array3::<usize>::from_shape_vec((1, 2, 3), vec![1, 1, 2, 1, 0, 1]).unwrap();

            assert_eq!(result.hist, expected)
        }
    }

    /// Test a histogram with filters is correctly constructed.
    #[test]
    fn test_hist_filter() {
        let times = Array1::from_vec(vec![500, 600, 1500, 2300, 2500, 2650]);
        let specs = Array1::from_vec(vec![0, 1, 0, 0, 0, 1]);
        let periods = Array1::zeros(6);
        let weights: [bool; 6] = [false, true, true, false, false, true];

        unsafe {
            let result = make_histogram(
                times,
                specs,
                2,
                &periods.slice(s![0..=5]),
                1,
                weights.into_iter().into(),
                0.,
                3.,
                3,
                1.,
                1e-3,
            );

            let expected =
                Array3::<usize>::from_shape_vec((1, 2, 3), vec![0, 1, 0, 1, 0, 1]).unwrap();

            assert_eq!(result.hist, expected)
        }
    }
}
