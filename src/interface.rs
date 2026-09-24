use anyhow::Result;
use numpy::ToPyArray;
use pyo3::prelude::{pyclass, pymethods, Bound};

use crate::batch_interface::{FilterIndex, PyHist};
use crate::{BatchData, NexusData};

use std::path::PathBuf;

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
    /// Create a new Data object to load and process muon event data.
    ///
    /// Parameters
    /// ----------
    /// filename: str
    ///     The filename of the muon nexus v2 file to open.
    /// n_spec: int
    ///     The number of detector spectra in the experiment.
    /// chunk_size: int
    ///     The chunk size to use when reading the dataset (defaults to 1,048,576).
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
    /// This processes the event dataset using the requested compute device,
    /// updating the cached histogram results.
    ///
    /// Parameters
    /// ----------
    /// device: str | None
    ///     Device to run on: 'auto' (default), 'cpu', or 'gpu'.
    ///     If None, uses the device set on the Data instance (defaults to 'auto').
    ///
    /// Returns
    /// -------
    /// Data
    ///     A Data object containing the resulting histogram
    ///     and number of events.
    #[pyo3(signature = (device=None))]
    pub fn calculate(&mut self, device: Option<&str>) -> Result<Data> {
        self.inner.calculate(device)?;
        Ok(self.clone())
    }

    /// Set the device preference for histogram calculations.
    /// This controls whether calculations run on the CPU or GPU.
    ///
    /// Parameters
    /// ----------
    /// device: str
    ///     The device to use. Must be one of 'auto', 'cpu', or 'gpu'.
    pub fn set_device(&mut self, device: &str) -> Result<()> {
        self.inner.set_device(device)
    }

    /// Get the current device preference.
    ///
    /// Returns
    /// -------
    /// str
    ///     The currently configured device preference ('auto', 'cpu', or 'gpu').
    pub fn get_device(&self) -> String {
        self.inner.get_device()
    }

    /// Force histograms to be recalculated even if the data hasn't changed.
    fn invalidate_cache(&mut self) {
        self.inner.invalidate_cache()
    }

    /// Set histogram settings.
    ///
    /// Parameters
    /// ----------
    /// min_time: float
    ///     The minimum time bound for the histogram in microseconds.
    /// max_time: float
    ///     The maximum time bound for the histogram in microseconds.
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
    /// autofill: bool
    ///     Whether to automatically fill the file with default values for
    ///     the missing meta-data (this is needed because the event data files
    ///     have mistakes/problems).
    ///     This allows the file to be read by Mantid even if the event file
    ///     is incomplete.
    /// ref_file: str
    ///     The reference file for the saved file. (must be a Nexus file)
    ///     Contains "correct" data that should be copied to the output file.
    ///     This is only need it the reference file needed is not the standard
    ///     muon nexus v2 file. The ref_file is generated from tools/make_default.py.
    #[pyo3(signature = (filename, autofill=true, ref_file = (PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("files/muon_ref.nxs")).display().to_string()))]
    fn save(&self, filename: String, autofill: bool, ref_file: String) -> Result<()> {
        self.inner
            .save(FilterIndex::Index(0), filename, autofill, ref_file.clone())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_device_settings() {
        let _guard = crate::test_utils::lock_hdf5_test();
        const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";
        let mut data = Data::new(TEST_FILE.to_string(), 64, 1048576).unwrap();
        assert_eq!(data.get_device(), "auto");

        data.set_device("cpu").unwrap();
        assert_eq!(data.get_device(), "cpu");

        data.set_device("gpu").unwrap();
        assert_eq!(data.get_device(), "gpu");

        assert!(data.set_device("hybrid").is_err());
        assert!(data.set_device("invalid").is_err());
    }

    #[test]
    fn test_data_calculate_with_device() {
        let _guard = crate::test_utils::lock_hdf5_test();
        const TEST_FILE: &str = "./tests/test_data/HIFI00195790.nxs";
        let mut data = Data::new(TEST_FILE.to_string(), 64, 1048576).unwrap();

        data.calculate(Some("cpu")).unwrap();
        assert_eq!(data.get_n_events(), 64147);

        if crate::gpu::GpuContext::get().is_some() {
            data.invalidate_cache();
            data.calculate(Some("gpu")).unwrap();
            assert_eq!(data.get_n_events(), 64147);
        }
    }
}
