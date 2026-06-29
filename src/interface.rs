use anyhow::Result;
use pyo3::prelude::{pyclass, pymethods};

use crate::data::NexusData;
use crate::filters::Filters;
use crate::stats::Histogram;

/// The main MNeuEventLib interface.
#[pyclass]
pub struct Data {
    dataset: NexusData,
    results: Histogram,
    filters: Filters
}

#[pymethods]
impl Data {
    /// Create a new Data object.
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    fn new(filename: String, n_spec: usize, chunk_size: usize) -> Result<Self> {
        let dataset = NexusData::new(filename, n_spec, chunk_size)?;
        let results = Histogram::new(0., 32.768, 2048);
        Ok(Data { dataset, results, filters: Filters::new() })
    }

    /// Calculate the histogram for the current data and filters.
    ///
    /// Returns
    /// -------
    /// Histogram
    ///     A Histogram object containing the resulting histogram
    ///     and number of events.
    fn calculate(&self) -> Result<Histogram> {
        let (hist, _) = self.results.calculate(&self.dataset, &self.filters)?;
        Ok(hist)
    }

    /// Set histogram settings.
    ///
    /// Parameters
    /// ----------
    /// min_time: float
    ///     The minimum time bound for the histogram.
    /// max_time: float
    ///     The maximum time bound for the histogram.
    /// n_bins: int
    ///     The number of bins to divide the time range into.
    fn set_histogram_settings(&mut self, min_time: f32, max_time: f32, n_bins: usize) {
        self.results = Histogram::new(min_time, max_time, n_bins)
    }
}
