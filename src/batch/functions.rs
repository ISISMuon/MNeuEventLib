//! Convenience functions for batch data.
use anyhow::{Error, Result};
use ndarray::Array1;
use pyo3::prelude::pyfunction;

use crate::batch::BatchData;

/// Add a time filter to every filter set, splitting the range from `start`
/// to `end` into evenly-spaced consecutive time filters, one per filter set.
///
/// Parameters
/// ----------
/// name: str
///     The name of the time filter. Must be unique within each filter set.
/// start: float
///     The start point of the first filter set's time filter.
/// end: float
///     The end point of the last filter set's time filter.
#[pyfunction]
pub fn time_linspace(name: String, start: f64, end: f64, num: usize) -> Result<BatchData> {
    let mut data = BatchData::empty(num);
    let array = Array1::linspace(start, end, num + 1);
    data.array_to_time_filters(name, array)?;
    Ok(data)
}

/// Add a time filter to every filter set, splitting the range from `start`
/// to `end` into geometrically (log)-spaced consecutive time filters, one per filter set.
///
/// Parameters
/// ----------
/// name: str
///     The name of the time filter. Must be unique within each filter set.
/// start: float
///     The start point of the first filter set's time filter.
/// end: float
///     The end point of the last filter set's time filter.
#[pyfunction]
pub fn time_geomspace(name: String, start: f64, end: f64, num: usize) -> Result<BatchData> {
    let mut data = BatchData::empty(num);
    let array = Array1::geomspace(start, end, num + 1)
        .ok_or(Error::msg("Invalid bounds for geometric spacing."))?;
    data.array_to_time_filters(name, array)?;
    Ok(data)
}

/// Add a time filter to every filter set, splitting the range starting at
/// `start` into consecutive time filters of width `step`, one per filter
/// set.
///
/// Parameters
/// ----------
/// name: str
///     The name of the time filter. Must be unique within each filter set.
/// start: float
///     The start point of the first filter set's time filter.
/// stop: float
///     The end point of the last filter set's time filter.
/// step: float
///     The width of each filter set's time filter.
#[pyfunction]
pub fn time_range(name: String, start: f64, end: f64, step: f64) -> Result<BatchData> {
    let array = Array1::range(start, end, step);
    let mut data = BatchData::empty(array.len() - 1);
    data.array_to_time_filters(name, array)?;
    Ok(data)
}

/// Add a sample log filter to every filter set, splitting the range from
/// `start` to `end` into evenly-spaced consecutive log filters, one per
/// filter set.
///
/// Parameters
/// ----------
/// name: str
///     The name of the log filter. Must be unique within each filter set.
/// log: str
///     The sample log in the data to which the filters apply.
/// start: float
///     The lower bound of the first filter set's log filter.
/// end: float
///     The upper bound of the last filter set's log filter.
#[pyfunction]
pub fn slog_linspace(
    name: String,
    log: String,
    start: f64,
    end: f64,
    num: usize,
) -> Result<BatchData> {
    let mut data = BatchData::empty(num);
    let array = Array1::linspace(start, end, num + 1);
    data.array_to_log_filters(name, log, array)?;
    Ok(data)
}

/// Add a sample log filter to every filter set, splitting the range from
/// `start` to `end` into geometrically (log)-spaced consecutive log filters,
/// one per filter set.
///
/// Parameters
/// ----------
/// name: str
///     The name of the log filter. Must be unique within each filter set.
/// log: str
///     The sample log in the data to which the filters apply.
/// start: float
///     The lower bound of the first filter set's log filter.
/// end: float
///     The upper bound of the last filter set's log filter.
#[pyfunction]
pub fn slog_geomspace(
    name: String,
    log: String,
    start: f64,
    end: f64,
    num: usize,
) -> Result<BatchData> {
    let mut data = BatchData::empty(num);
    let array = Array1::geomspace(start, end, num + 1)
        .ok_or(Error::msg("Invalid bounds for geometric spacing."))?;
    data.array_to_log_filters(name, log, array)?;
    Ok(data)
}

/// Add a sample log filter to every filter set, splitting the range
/// starting at `start` into consecutive log filters of width `step`, one
/// per filter set.
///
/// Parameters
/// ----------
/// name: str
///     The name of the log filter. Must be unique within each filter set.
/// log: str
///     The sample log in the data to which the filters apply.
/// start: float
///     The start point of the first filter set's log filter.
/// stop: float
///     The end point of the last filter set's log filter.
/// step: float
///     The width of each filter set's log filter.
#[pyfunction]
pub fn slog_range(name: String, log: String, start: f64, end: f64, step: f64) -> Result<BatchData> {
    let array = Array1::range(start, end, step);
    let mut data = BatchData::empty(array.len() - 1);
    data.array_to_log_filters(name, log, array)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consts::ToNanoseconds;

    /// The bounds of every filter set's single time filter, in order of
    /// filter set.
    fn time_bounds(data: &BatchData) -> Vec<(u64, u64)> {
        data.filters
            .iter()
            .map(|filters| {
                let (starts, ends) = filters.get_time_filter_times();
                assert_eq!(starts.len(), 1, "expected exactly one time filter");
                (starts[0], ends[0])
            })
            .collect()
    }

    /// The bounds of every filter set's single log filter named `name`, in
    /// order of filter set.
    fn log_bounds(data: &BatchData, name: &str) -> Vec<(f64, f64)> {
        data.filters
            .iter()
            .map(|filters| {
                assert_eq!(filters.sample_log_filters.len(), 1);
                let filter = &filters.sample_log_filters[name];
                (filter.lower.unwrap(), filter.upper.unwrap())
            })
            .collect()
    }

    /// Check two sequences of bounds agree to within a small tolerance.
    fn assert_close(actual: Vec<(f64, f64)>, expected: Vec<(f64, f64)>) {
        assert_eq!(actual.len(), expected.len());
        for ((a_lo, a_hi), (e_lo, e_hi)) in actual.into_iter().zip(expected) {
            assert!((a_lo - e_lo).abs() < 1e-9, "{a_lo} != {e_lo}");
            assert!((a_hi - e_hi).abs() < 1e-9, "{a_hi} != {e_hi}");
        }
    }

    /// Check two sequences of times in ns agree to within a nanosecond
    fn assert_close_ns(actual: Vec<(u64, u64)>, expected: Vec<(u64, u64)>) {
        assert_eq!(actual.len(), expected.len());
        for ((a_lo, a_hi), (e_lo, e_hi)) in actual.into_iter().zip(expected) {
            assert!(a_lo.abs_diff(e_lo) <= 1, "{a_lo} != {e_lo}");
            assert!(a_hi.abs_diff(e_hi) <= 1, "{a_hi} != {e_hi}");
        }
    }

    /// time_linspace should give `num` filter sets, each holding one time
    /// filter covering a consecutive, evenly-spaced slice of the range.
    #[test]
    fn test_time_linspace() {
        let data = time_linspace("f".to_string(), 0., 4., 4).unwrap();

        assert_eq!(data.__len__(), 4);
        assert_eq!(
            time_bounds(&data),
            vec![
                (0f64.to_ns(), 1f64.to_ns()),
                (1f64.to_ns(), 2f64.to_ns()),
                (2f64.to_ns(), 3f64.to_ns()),
                (3f64.to_ns(), 4f64.to_ns()),
            ]
        );
    }

    /// time_geomspace should give `num` filter sets covering consecutive,
    /// geometrically-spaced slices of the range.
    #[test]
    fn test_time_geomspace() {
        let data = time_geomspace("f".to_string(), 1., 16., 4).unwrap();

        assert_eq!(data.__len__(), 4);
        assert_close_ns(
            time_bounds(&data),
            vec![
                (1f64.to_ns(), 2f64.to_ns()),
                (2f64.to_ns(), 4f64.to_ns()),
                (4f64.to_ns(), 8f64.to_ns()),
                (8f64.to_ns(), 16f64.to_ns()),
            ],
        );
    }

    /// time_range should give one filter set per step, each `step` wide.
    #[test]
    fn test_time_range() {
        let data = time_range("f".to_string(), 0., 2., 0.5).unwrap();

        // the range [0, 2) in steps of 0.5 is [0, 0.5, 1, 1.5], bounding
        // three filter sets
        assert_eq!(data.__len__(), 3);
        assert_eq!(
            time_bounds(&data),
            vec![
                (0f64.to_ns(), 0.5f64.to_ns()),
                (0.5f64.to_ns(), 1f64.to_ns()),
                (1f64.to_ns(), 1.5f64.to_ns()),
            ]
        );
    }

    /// slog_linspace should give `num` filter sets, each holding one log
    /// filter covering a consecutive, evenly-spaced slice of the range.
    #[test]
    fn test_slog_linspace() {
        let data = slog_linspace("lf".to_string(), "temp".to_string(), 0., 4., 4).unwrap();

        assert_eq!(data.__len__(), 4);
        assert_close(
            log_bounds(&data, "lf"),
            vec![(0., 1.), (1., 2.), (2., 3.), (3., 4.)],
        );
    }

    /// slog_geomspace should give `num` filter sets covering consecutive,
    /// geometrically-spaced slices of the range.
    #[test]
    fn test_slog_geomspace() {
        let data = slog_geomspace("lf".to_string(), "temp".to_string(), 1., 16., 4).unwrap();

        assert_eq!(data.__len__(), 4);
        assert_close(
            log_bounds(&data, "lf"),
            vec![(1., 2.), (2., 4.), (4., 8.), (8., 16.)],
        );
    }

    /// slog_range should give one filter set per step, each `step` wide.
    #[test]
    fn test_slog_range() {
        let data = slog_range("lf".to_string(), "temp".to_string(), 0., 2., 0.5).unwrap();

        // the range [0, 2) in steps of 0.5 is [0, 0.5, 1, 1.5], bounding
        // three filter sets
        assert_eq!(data.__len__(), 3);
        assert_close(
            log_bounds(&data, "lf"),
            vec![(0., 0.5), (0.5, 1.), (1., 1.5)],
        );
    }
}
