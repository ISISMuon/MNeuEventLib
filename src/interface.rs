use anyhow::{Error, Result};
use pyo3::prelude::{pyclass, pymethods};

use crate::data::save::sanitise::nexus_data::{save_default, get_p_info};
use crate::data::{NexusData, SaveFile, WiMDAFile};
use crate::filters::Filters;
use crate::stats::Histogram;

use std::path::PathBuf;

/// The main MNeuEventLib interface.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Data {
    #[pyo3(get)]
    pub dataset: NexusData,
    pub results: Histogram,
    pub filters: Filters,
    data_changed: bool, // whether data has changed since last calculation
}

#[pymethods]
impl Data {
    /// Create a new Data object.
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    pub fn new(filename: String, n_spec: usize, chunk_size: usize) -> Result<Self> {
        let dataset = NexusData::new(filename, n_spec, chunk_size)?;
        let results = Histogram::new(0., 32.768, 2048);
        Ok(Data {
            dataset,
            results,
            filters: Filters::new(),
            data_changed: true,
        })
    }

    /// Calculate the histogram for the current data and filters.
    ///
    /// Returns
    /// -------
    /// Histogram
    ///     A Histogram object containing the resulting histogram
    ///     and number of events.
    pub fn calculate(&mut self) -> Result<Data> {
        if self.data_changed {
            self.data_changed = false;
            let results = self.results.calculate(&self.dataset, &self.filters)?;
            self.results = results.clone();
            Ok(self.clone())
        } else {
            // if data hasn't changed, just return the existing saved results
            Ok(self.clone())
        }
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
        if n_bins == 0 {
            return Err(Error::msg("n_bins must be greater than 0."));
        }
        if !min_time.is_finite() || !max_time.is_finite() {
            return Err(Error::msg("min_time and max_time must be finite."));
        }
        if max_time <= min_time {
            return Err(Error::msg("max_time must be greater than min_time."));
        }
        self.data_changed = true;
        self.results = Histogram::new(min_time, max_time, n_bins);
        Ok(())
    }

    /// Set the type for the time filters.
    ///
    /// Parameters
    /// ----------
    /// filter_type: str
    ///     The type for the time filters. Must be 'exclude' or 'include'.
    fn set_time_type(&mut self, filter_type: String) -> Result<()> {
        self.data_changed = true;
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
    pub fn add_time_filter(&mut self, name: String, start: f64, end: f64) -> Result<()> {
        self.data_changed = true;
        self.filters.add_time_filter(name, start, end)
    }

    /// Remove a time filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the time filter to remove.
    fn remove_time_filter(&mut self, name: String) -> Result<()> {
        self.data_changed = true;
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
    pub fn add_log_filter(
        &mut self,
        name: String,
        log: String,
        lower: f64,
        upper: f64,
    ) -> Result<()> {
        self.data_changed = true;
        self.filters
            .add_log_filter(name, log, Some(lower), Some(upper))
    }

    /// Remove a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// name: str
    ///     The name of the log filter to remove.
    fn remove_log_filter(&mut self, name: String) -> Result<()> {
        self.data_changed = true;
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
        self.data_changed = true;
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
        self.data_changed = true;
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
        self.data_changed = true;
        self.filters.set_amp(detector, amp)
    }

    /// Set an amplitude filter for all detectors that don't have one defined.
    ///
    /// Parameters
    /// ----------
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    fn set_amps_baseline(&mut self, amp: f64) {
        self.data_changed = true;
        self.filters.set_amps_baseline(amp)
    }

    /// Save to a file.
    ///
    /// Parameters
    /// ----------
    /// filename: str
    ///     The filename for the saved file.
    fn save(&self, filename: String) -> Result<()> {
        let wimda_file = WiMDAFile::new(self)?;
        wimda_file.save(filename, &self.dataset.file)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "{}\n\n{}\n\n{}",
            self.dataset.__repr__(),
            self.filters.__repr__(),
            self.results.__repr__()
        )
    }

    /// Save to a Nexus version 2 file that is compatable with
    /// Mantid using provided reference file for data.
    /// This is needed because the event data files has mistakes/problems.
    /// 
    /// Parameters
    /// ----------
    /// filename: str
    ///     The filename for the saved file.
    /// ref_file: str
    ///     The reference file for the saved file. (must be a Nexus file)
    ///     Contains "correct" data that should be copied to the output file.
    ///     This can be generated from tools/make_default.py
    ///
    /// Returns
    /// -------
    /// Result<()>
    ///     Ok(()) if the file is saved successfully
    ///     Err(anyhow::Error) if the file cannot be saved
    #[pyo3(signature = (filename, ref_file = (PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("files/muon_ref.nxs")).display().to_string()))]
    pub fn save_nexus(&self, filename: String, ref_file: String) -> Result<()> {
        println!("mooo {} " , ref_file);    

hdf5::silence_errors(false);
        // 1. Save using the existing WiMDA save logic to `filename`
        let wimda_file = WiMDAFile::new(self)?;
        wimda_file.save(filename.clone(), &self.dataset.file)?;
        // 2. Read p_info from input file
        let (periods, dwell) = get_p_info(&self.dataset.filename)?;

        // 3. Setup shapes map
        let mut shapes = std::collections::HashMap::new();
        let n = self.dataset.n_spec;
        println!("checking {}", n);
        shapes.insert("N".to_string(), n);
        shapes.insert("P".to_string(), periods);
        shapes.insert("NP".to_string(), n * periods);
        shapes.insert("PD".to_string(), periods + dwell);
        shapes.insert("NPD".to_string(), n * (periods + dwell));

        // 4. Run save_default to merge/copy from ref_file
        save_default(&filename, &ref_file, &shapes)?;

        Ok(())
    }
}
