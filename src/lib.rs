use pyo3::prelude::*;

mod data;
use data::Data;

/// A Python module implemented in Rust.
#[pymodule]
fn mneueventlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Data>()?;
    Ok(())
}
