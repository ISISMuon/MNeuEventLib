//! Functionality for batch-processing data.
mod interface;
pub use interface::{BatchData, FilterIndex, PyHist};
mod functions;
pub use functions::{
    slog_geomspace, slog_linspace, slog_range, time_geomspace, time_linspace, time_range,
};
