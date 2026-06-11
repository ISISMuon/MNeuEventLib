#![allow(non_snake_case)]
use pyo3::prelude::*;

mod data;
use data::NexusData;
mod stats;
use stats::Histogram;
mod filters;
use filters::Filters;
mod consts;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
fn MNeuEventLib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NexusData>()?;
    m.add_class::<Histogram>()?;
    m.add_class::<Filters>()?;
    Ok(())
}
