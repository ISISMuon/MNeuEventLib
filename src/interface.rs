use anyhow::Result;
use pyo3::prelude::{pyclass, pymethods};

use crate::data::NexusData;
use crate::filters::Filters;
use crate::stats::Histogram;

/// The main MNeuEventLib interface.
#[pyclass]
pub struct Data {
    #[pyo3(get)]
    dataset: NexusData,
    results: Histogram,
    filters: Filters,
}

#[pymethods]
impl Data {
    /// Create a new Data object.
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    fn new(filename: String, n_spec: usize, chunk_size: usize) -> Result<Self> {
        let dataset = NexusData::new(filename, n_spec, chunk_size)?;
        let results = Histogram::new(0., 32.768, 2048);
        Ok(Data {
            dataset,
            results,
            filters: Filters::new(),
        })
    }

    /// Calculate the histogram for the current data and filters.
    ///
    /// Returns
    /// -------
    /// Histogram
    ///     A Histogram object containing the resulting histogram
    ///     and number of events.
    fn calculate(&self) -> Result<Histogram> {
        self.results.calculate(&self.dataset, &self.filters)
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

    /// Set the type for the time filters.
    ///
    /// Parameters
    /// ----------
    /// filter_type: str
    ///     The type for the time filters. Must be 'exclude' or 'include'.
    fn set_time_type(&mut self, filter_type: String) -> Result<()> {
        self.filters.set_time_type(filter_type)
    }

    /// Add a time filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the time filter. Must be unique.
    /// start: float
    ///     The start point for the time filter.
    /// end: float
    ///     The end point for the time filter.
    fn add_time_filter(&mut self, name: String, start: f64, end: f64) -> Result<()> {
        self.filters.add_time_filter(name, start, end)
    }

    /// Remove a time filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the time filter to remove.
    fn remove_time_filter(&mut self, name: String) -> Result<()> {
        self.filters.remove_time_filter(name)
    }

    /// Add a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the log filter. Must be unique.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// lower: float
    ///     The lower bound for the log filter.
    /// upper: float
    ///     The upper bound for the log filter.
    fn add_log_filter(&mut self, name: String, log: String, lower: f64, upper: f64) -> Result<()> {
        self.filters.add_log_filter(name, log, lower, upper)
    }

    /// Remove a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the log filter to remove.
    fn remove_log_filter(&mut self, name: String) -> Result<()> {
        self.filters.remove_log_filter(name)
    }

    /// Add a sample log filter for all data above a certain value.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the log filter. Must be unique.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// lower: float
    ///     The lower bound for the log filter.
    fn add_log_filter_above(&mut self, name: String, log: String, lower: f64) -> Result<()> {
        self.filters.add_log_filter_above(name, log, lower)
    }

    /// Add a sample log filter for all data below a certain value.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the log filter. Must be unique.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// upper: float
    ///     The upper bound for the log filter.
    fn add_log_filter_below(&mut self, name: String, log: String, upper: f64) -> Result<()> {
        self.filters.add_log_filter_below(name, log, upper)
    }

    /// Set the amplitude filter for a detector.
    ///
    /// Parameters
    /// ----------
    /// detector: int
    ///     The detector to set a filter for.
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    fn set_amp(&mut self, detector: usize, amp: f64) {
        self.filters.set_amp(detector, amp)
    }

    /// Set an amplitude filter for all detectors that don't have one defined.
    ///
    /// Parameters
    /// ----------
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    fn set_amps_baseline(&mut self, amp: f64) {
        self.filters.set_amps_baseline(amp)
    }
}
