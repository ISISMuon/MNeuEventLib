use pyo3::prelude::*;

mod data;
use data::NexusData;
mod stats;
use stats::Histogram;
mod filters;

/// A Python module implemented in Rust.
#[pymodule]
fn mneueventlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NexusData>()?;
    m.add_class::<Histogram>()?;
    Ok(())
}
