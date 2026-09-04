//! A module for handling input data.

mod nexus_data;
pub use nexus_data::NexusData;
mod sample_logs;
pub use sample_logs::SampleLog;
mod frame_data;
pub use frame_data::FrameData;
pub mod save;
pub use save::{SaveFile, WiMDAFile};

// we use ValueLog in tests to directly create sample logs from arrays
#[allow(unused_imports)]
pub(crate) use sample_logs::ValueLog;
