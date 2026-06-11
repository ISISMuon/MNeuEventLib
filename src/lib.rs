use pyo3::prelude::*;

mod data;
use data::NexusData;

/// A Python module implemented in Rust.
#[pymodule]
fn mneueventlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NexusData>()?;
    Ok(())
}
