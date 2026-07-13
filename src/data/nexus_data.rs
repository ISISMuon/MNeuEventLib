use std::cmp::min;
use std::collections::HashMap;
use std::path::Path;

use anyhow::{Error, Result};
use hdf5::{Dataset, File, Group};
use ndarray::{s, Array1};
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, Bound, PyDictMethods, Python};
use pyo3::types::PyDict;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::data::SampleLog;

/// Class for storing a Nexus event file.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct NexusData {
    pub file: File,
    pub filename: String,
    pub specs: Dataset,
    pub times: Dataset,
    pub amps: Dataset,
    pub frames: Dataset,
    pub frame_times: Dataset,
    pub periods: Dataset,
    pub sample_logs: Group,
    pub sample_log_names: Vec<String>,
    pub n_events: usize, // the total number of events
    pub n_frames: usize,
    pub n_spec: usize,     // the number of detectors
    pub chunk_size: usize, // the size of the data chunks
}

#[pymethods]
impl NexusData {
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    pub fn new(filename: String, n_spec: usize, chunk_size: usize) -> Result<Self> {
        let path = Path::new(&filename);
        load_data(path, n_spec, chunk_size)
    }

    /// used for testing
    fn get_frame_times<'py>(slf: &Bound<'py, NexusData>) -> Bound<'py, PyArray1<u32>> {
        let py = slf.py();
        slf.borrow().frame_times.read_1d().unwrap().to_pyarray(py)
    }

    // Python wrapper for get_sample_log, to avoid having to make SampleLog a pyclass
    /// Retrieve the data for a sample log.
    #[pyo3(name = "get_sample_log")]
    fn get_sample_log_py<'py>(
        &self,
        log_name: String,
        py: Python<'py>,
    ) -> Result<Bound<'py, PyDict>> {
        let sample_log = self.get_sample_log(&log_name)?;

        let output = PyDict::new(py);
        // PyDict is dynamically typed (as it's a Python object),
        // but we need to unpack the type of the SampleLog anyway to make the Rust compiler happy
        match sample_log {
            SampleLog::Float(log) => {
                output.set_item("time".to_string(), log.time.to_pyarray(py))?;
                output.set_item("value".to_string(), log.value.to_pyarray(py))?;
            }
            SampleLog::Int(log) => {
                output.set_item("time".to_string(), log.time.to_pyarray(py))?;
                output.set_item("value".to_string(), log.value.to_pyarray(py))?;
            }
        };
        Ok(output)
    }

    /// Get a histogram of the amplitudes and the maximum height.
    #[pyo3(name = "get_amp_histogram", signature = (max_height=None, n_bins=10))]
    fn get_amp_histogram_py<'py>(
        &self,
        py: Python<'py>,
        max_height: Option<f64>,
        n_bins: usize,
    ) -> Result<(Bound<'py, PyArray1<usize>>, f64)> {
        let (results, max) = self.get_amp_histogram(max_height, n_bins)?;
        Ok((results.to_pyarray(py), max))
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
        SampleLog::new(log_name, time, value)
    }

    /// Get the value logs associated with a list of sample log names.
    pub fn get_sample_logs(&self, log_names: Vec<String>) -> Result<HashMap<String, SampleLog>> {
        log_names
            .into_par_iter()
            .map(|name| self.get_sample_log(&name).map(|log| (name, log)))
            .collect()
    }

    /// Get the histogram of amplitudes and the max height.
    fn get_amp_histogram(
        &self,
        max_height: Option<f64>,
        n_bins: usize,
    ) -> Result<(Array1<usize>, f64)> {
        let max = match max_height {
            Some(height) => height,
            None => self.get_dataset_max(&self.amps)?,
        };
        let width = max / n_bins as f64;

        // parallel iterate over chunks
        let results = Array1::from_iter(
            (0..self.n_events)
                .into_par_iter()
                .step_by(self.chunk_size)
                // get the amps for each chunk
                .map(|start| -> Array1<f64> {
                    let end = min(start + self.chunk_size, self.n_events);
                    let array_slice = s![start..end];
                    self.amps
                        .read_slice_1d(array_slice)
                        .expect("Failed to read amplitude data.")
                })
                // bin amps for each chunk
                .map(|amps| {
                    let mut array = Array1::zeros(n_bins);
                    for amp in amps {
                        // anything over the max is put in the last bin
                        if amp >= max {
                            array[n_bins - 1] += 1
                        } else {
                            let bin = (amp / width).floor() as usize;
                            array[bin] += 1
                        }
                    }
                    array
                })
                // combine chunks
                .reduce(
                    // rayon's reduce requires an identity value
                    || Array1::zeros(n_bins),
                    |mut acc, r| {
                        acc += &r;
                        acc
                    },
                ),
        );
        Ok((results, max))
    }

    /// Get the maximum value of a dataset.
    fn get_dataset_max(&self, dataset: &Dataset) -> Result<f64> {
        Ok((0..self.n_events)
            .into_par_iter()
            .step_by(self.chunk_size)
            .map(|start| -> f64 {
                let end = min(start + self.chunk_size, self.n_events);
                let array_slice = s![start..end];
                let array: Array1<f64> = dataset
                    .read_slice_1d(array_slice)
                    .expect("Failed to read amplitude data.");
                array
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap())
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
    let n_frames = frames.size();

    Ok(NexusData {
        file,
        filename: filename.to_str().unwrap().to_string(),
        specs,
        times,
        amps,
        frames,
        frame_times,
        periods,
        sample_logs,
        sample_log_names,
        n_events,
        n_frames,
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
        assert!(matches!(value_log, SampleLog::F32(_)))
    }

    /// Test that a non-real sample log throws an error.
    #[test]
    fn test_load_sample_log_not_found() {
        let data = test_data();

        let log = data.get_sample_log(&"Lunch".to_string());
        assert!(log.is_err())
    }

    /// Test that an amplitude histogram is successfully created for some data.
    #[test]
    fn test_make_amp_histogram() {
        let data = test_data();

        let result = data.get_amp_histogram(None, 10);
        assert!(result.is_ok())
    }
}
