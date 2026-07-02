//! # MNeuEventLib
//!
//! MNeuEventLib is a Python package (written in Rust) for processing ISIS event data into a NeXuS
//! version 2 histogram file. This processing may include filtering the events based on:
//!
//! * the time they occurred;
//! * on the values of auxiliary logs such as sample logs, warnings, or vetos;
//! * or on a high-pass filter of event amplitudes per detector.
//!
//! It is primarily for muon event data, and the histogram files are
//! currently only compatible with [WiMDA](https://shadow.nd.rl.ac.uk/wimda/).
#![allow(non_snake_case)]
use pyo3::prelude::*;

mod data;
use data::NexusData;
mod stats;
use stats::Histogram;
mod filters;
use filters::Filters;
mod interface;
use interface::Data;
mod consts;
mod utils;

/// A Python module implemented in Rust.
#[pymodule]
fn MNeuEventLib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NexusData>()?;
    m.add_class::<Histogram>()?;
    m.add_class::<Filters>()?;
    m.add_class::<Data>()?;
    Ok(())
}
