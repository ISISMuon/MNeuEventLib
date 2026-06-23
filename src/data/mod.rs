//! A module for handling input data.

mod nexus_data;
pub use nexus_data::NexusData;
mod sample_logs;
pub use sample_logs::SampleLog;
mod rle_array;
use rle_array::{RLEArray, RLEArraySlice};

// we use ValueLog in tests to directly create sample logs from arrays
#[allow(unused_imports)]
pub(crate) use sample_logs::ValueLog;
