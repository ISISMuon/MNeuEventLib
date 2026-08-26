use anyhow::Result;
use numpy::ToPyArray;
use pyo3::prelude::{pyclass, pymethods, Bound};

use crate::batch_interface::{FilterIndex, PyHist};
use crate::{BatchData, NexusData};

/// The main MNeuEventLib interface.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Data {
    // internally, to avoid code duplication,
    // this is treated as a BatchData with 1 batch
    inner: BatchData,
}

#[pymethods]
impl Data {
    /// Create a new Data object.
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    pub fn new(filename: String, n_spec: usize, chunk_size: usize) -> Result<Self> {
        Ok(Data {
            inner: BatchData::new(filename, n_spec, 1, chunk_size)?,
        })
    }

    #[getter]
    fn dataset(&self) -> NexusData {
        self.inner.dataset.clone()
    }

    /// Calculate the histogram for the current data and filters.
    ///
    /// Returns
    /// -------
    /// Histogram
    ///     A Histogram object containing the resulting histogram
    ///     and number of events.
    pub fn calculate(&mut self) -> Result<Data> {
        self.inner.calculate()?;
        Ok(self.clone())
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
    fn set_histogram_settings(
        &mut self,
        min_time: f32,
        max_time: f32,
        n_bins: usize,
    ) -> Result<()> {
        self.inner
            .set_histogram_settings(FilterIndex::Index(0), min_time, max_time, n_bins)
    }

    /// Set the type for the time filters.
    ///
    /// Parameters
    /// ----------
    /// filter_type: str
    ///     The type for the time filters. Must be 'exclude' or 'include'.
    fn set_time_type(&mut self, filter_type: String) -> Result<()> {
        self.inner.set_time_type(FilterIndex::Index(0), filter_type)
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
    pub fn add_time_filter(&mut self, name: String, start: f64, end: f64) -> Result<()> {
        self.inner
            .add_time_filter(FilterIndex::Index(0), name, start, end)
    }

    /// Remove a time filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the time filter to remove.
    fn remove_time_filter(&mut self, name: String) -> Result<()> {
        self.inner.remove_time_filter(FilterIndex::Index(0), name)
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
    pub fn add_log_filter(
        &mut self,
        name: String,
        log: String,
        lower: f64,
        upper: f64,
    ) -> Result<()> {
        self.inner
            .add_log_filter(FilterIndex::Index(0), name, log, lower, upper)
    }

    /// Remove a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the log filter to remove.
    fn remove_log_filter(&mut self, name: String) -> Result<()> {
        self.inner.remove_log_filter(FilterIndex::Index(0), name)
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
        self.inner
            .add_log_filter_above(FilterIndex::Index(0), name, log, lower)
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
        self.inner
            .add_log_filter_below(FilterIndex::Index(0), name, log, upper)
    }

    /// Set the amplitude filter for a detector.
    ///
    /// Parameters
    /// ----------
    /// detector: int
    ///     The detector to set a filter for.
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    fn set_amp(&mut self, detector: usize, amp: f64) -> Result<()> {
        self.inner.set_amp(FilterIndex::Index(0), detector, amp)
    }

    /// Set an amplitude filter for all detectors that don't have one defined.
    ///
    /// Parameters
    /// ----------
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    fn set_amps_baseline(&mut self, amp: f64) -> Result<()> {
        self.inner.set_amps_baseline(FilterIndex::Index(0), amp)
    }

    /// Save to a file.
    ///
    /// Parameters
    /// ----------
    /// filename: str
    ///     The filename for the saved file.
    fn save(&self, filename: String) -> Result<()> {
        self.inner.save(FilterIndex::Index(0), filename)
    }

    /// Get the calculated histogram.
    fn get_histogram<'py>(slf: &Bound<'py, Data>) -> PyHist<'py> {
        let py = slf.py();
        slf.borrow().inner.results[0].hist.to_pyarray(py)
    }

    /// Get the number of events.
    fn get_n_events(&self) -> usize {
        self.inner.results[0].n
    }

    fn __repr__(&self) -> String {
        format!(
            "{}\n\n{}\n\n{}",
            self.inner.dataset.__repr__(),
            self.inner.filters[0].__repr__(),
            self.inner.results[0].__repr__()
        )
    }
}
