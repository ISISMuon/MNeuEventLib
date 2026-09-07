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
pub fn log_linspace(
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
pub fn log_geomspace(
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
pub fn log_range(name: String, log: String, start: f64, end: f64, step: f64) -> Result<BatchData> {
    let array = Array1::range(start, end, step);
    let mut data = BatchData::empty(array.len() - 1);
    data.array_to_log_filters(name, log, array)?;
    Ok(data)
}
