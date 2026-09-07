//! Functionality for batch-processing data.
mod interface;
pub use interface::{BatchData, FilterIndex, PyHist};
mod functions;
pub use functions::{
    log_geomspace, log_linspace, log_range, time_geomspace, time_linspace, time_range,
};
