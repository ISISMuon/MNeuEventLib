use std::collections::HashMap;
use std::path::Path;

use anyhow::{Error, Result};
use hdf5::types::TypeDescriptor;
use hdf5::{Dataset, File, Group};
use ndarray::Array1;
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, Bound};

use crate::data::{SampleLog, ValueLog};

/// Class for storing a Nexus event file.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct NexusData {
    pub file: String,
    pub specs: Dataset,
    pub times: Dataset,
    pub amps: Dataset,
    pub frames: Dataset,
    pub frame_times: Dataset,
    pub periods: Dataset,
    pub sample_logs: Group,
    pub sample_log_names: Vec<String>,
    pub n_events: usize,   // the total number of events
    pub n_spec: usize,     // the number of detectors
    pub chunk_size: usize, // the size of the data chunks
}

#[pymethods]
impl NexusData {
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    fn new(filename: String, n_spec: usize, chunk_size: usize) -> Result<Self> {
        let path = Path::new(&filename);
        load_data(path, n_spec, chunk_size)
    }

    /// used for testing
    fn get_frame_times<'py>(slf: &Bound<'py, NexusData>) -> Bound<'py, PyArray1<u32>> {
        let py = slf.py();
        slf.borrow().frame_times.read_1d().unwrap().to_pyarray(py)
    }
}

impl NexusData {
    /// Retreve the data for a sample log.
    pub fn get_sample_log(&self, log_name: &String) -> Result<SampleLog> {
        let log = match self.sample_logs.group(log_name) {
            Ok(group) => group,
            Err(_) => return Err(Error::msg(format!("Sample log {} not found!", log_name))),
        };
        let log_data = log.group("value_log")?;

        let time: Array1<f64> = log_data.dataset("time")?.read_1d()?;
        let value: Dataset = log_data.dataset("value")?;
        match value.dtype()?.to_descriptor()? {
            TypeDescriptor::Integer(_) => Ok(SampleLog::Int(ValueLog::<i32> {
                time,
                value: value.read_1d()?,
            })),
            TypeDescriptor::Float(_) => Ok(SampleLog::Float(ValueLog::<f64> {
                time,
                value: value.read_1d()?,
            })),
            other_type => Err(Error::msg(format!(
                "Sample log type {other_type} for log {log_name} is not supported.
                Supported types are Integer and Float."
            ))),
        }
    }

    /// Get the value logs associated with a list of sample log names.
    pub fn get_sample_logs(&self, log_names: Vec<String>) -> Result<HashMap<String, SampleLog>> {
        log_names
            .into_iter()
            .map(|name| self.get_sample_log(&name).map(|log| (name, log)))
            .collect()
    }
}

/// Load the data in a Nexus event data file into a NexusDataobject.
///
/// Parameters
/// ----------
/// filename: &Path
///     The path to the file.
/// n_spec: usize
///     The number of detectors used.
/// chunk_size: usize
///     The size of the HDF5 chunks in the data (usually 1048576)
///
/// Returns
/// -------
/// Data
///     A data object containing the relevant datasets.
///
fn load_data(filename: &Path, n_spec: usize, chunk_size: usize) -> Result<NexusData> {
    let file = File::open(filename)?;
    let data = file.group("raw_data_1")?;
    let events = data.group("detector_1_events")?;

    let specs = events.dataset("event_id")?;
    let times = events.dataset("event_time_offset")?;
    let amps = events.dataset("pulse_height")?;

    let frames = events.dataset("event_index")?;
    let frame_times = events.dataset("event_time_zero")?;
    let periods = events.dataset("period_number")?;

    let sample_logs = data.group("selog")?;
    let sample_log_names = sample_logs.member_names()?;

    let n_events = specs.size();

    Ok(NexusData {
        file: filename.to_str().unwrap().to_string(),
        specs,
        times,
        amps,
        frames,
        frame_times,
        periods,
        sample_logs,
        sample_log_names,
        n_events,
        n_spec,
        chunk_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data() -> NexusData {
        let path = Path::new("./tests/test_data/HIFI00195790.nxs");
        load_data(path, 64, 1048576).unwrap()
    }

    /// Test the program creates data when you load an existing file.
    #[test]
    fn test_file_load() {
        let path = Path::new("./tests/test_data/HIFI00195790.nxs");
        let data = load_data(path, 64, 1048576);

        assert!(data.is_ok())
    }

    /// Test the program returns an error when the file doesn't exist.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("./mysterious/unreal/fake_data.nxs");
        let data = load_data(path, 64, 1048576);
        assert!(data.is_err())
    }

    /// Test the sample log names are correctly loaded in.
    #[test]
    fn test_sample_log_names() {
        let data = test_data();

        assert_eq!(data.sample_log_names, vec!("Temp".to_string()))
    }

    /// Test that an existing sample log can be read successfully.
    #[test]
    fn test_load_sample_log() {
        let data = test_data();

        let log = data.get_sample_log(&"Temp".to_string());
        assert!(log.is_ok());

        let value_log = log.unwrap();
        assert!(matches!(value_log, SampleLog::Float(_)))
    }

    /// Test that a non-real sample log throws an error.
    #[test]
    fn test_load_sample_log_not_found() {
        let data = test_data();

        let log = data.get_sample_log(&"Lunch".to_string());
        assert!(log.is_err())
    }
}
