use std::cmp::min;
use std::collections::{HashMap, HashSet};
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

        // we need to use pattern matching to turn each log into a PyDict
        // and PyDict is dynamically typed but we need to unpack the type anyway
        // to make the Rust compiler happy
        macro_rules! pydict_for_type {
            ( $($type:path),+ ) => {
                match sample_log {
                    $(
                        $type(log) => {
                            output.set_item("name".to_string(), log.name)?;
                            output.set_item("time".to_string(), log.time.to_pyarray(py))?;
                            output.set_item("value".to_string(), log.value.to_pyarray(py))?;
                            output.set_item("unit".to_string(), log.unit)?;
                        }
                    )+
                }
            }
        }

        pydict_for_type! {
            SampleLog::I8,
            SampleLog::I16,
            SampleLog::I32,
            SampleLog::I64,
            SampleLog::U8,
            SampleLog::U16,
            SampleLog::U32,
            SampleLog::U64,
            SampleLog::F32,
            SampleLog::F64
        }
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

    pub fn __repr__(&self) -> String {
        "Dataset: ".to_owned() + &self.filename.clone()
    }
}

impl NexusData {
    /// Retreve the data for a sample log.
    pub fn new(filename: String, n_spec: usize, chunk_size: usize) -> Result<Self> {
        let path = Path::new(&filename);
        load_data(path, n_spec, chunk_size)
    }
    pub fn get_sample_log(&self, log_name: &String) -> Result<SampleLog> {
        let log = match self.sample_logs.group(log_name) {
            Ok(group) => group,
            Err(_) => return Err(Error::msg(format!("Sample log {log_name} not found!"))),
        };
        let log_data = log.group("value_log")?;

        let time: Array1<f64> = log_data.dataset("time")?.read_1d()?;
        let value: Dataset = log_data.dataset("value")?;
        SampleLog::new(log_name, time, value)
    }

    /// Get the value logs associated with a list of sample log names.
    pub fn get_sample_logs(
        &self,
        log_names: HashSet<String>,
    ) -> Result<HashMap<String, SampleLog>> {
        log_names
            .into_par_iter()
            .map(|name| self.get_sample_log(&name).map(|log| (name, log)))
            .collect()
    }

    /// Get the histogram of amplitudes and the max height.
    #[inline(always)]
    fn get_amp_histogram(
        &self,
        max_height: Option<f64>,
        n_bins: usize,
    ) -> Result<(Array1<usize>, f64)> {
        let max = match max_height {
            Some(height) if height.is_finite() => height,
            Some(_) => return Err(Error::msg("max_height must be finite.")),
            None => self.get_dataset_max(&self.amps)?,
        };
        if n_bins == 0 {
            return Err(Error::msg("n_bins must be greater than 0."));
        }
        let width = max / n_bins as f64;
        let n_amps = self.amps.size();

        // parallel iterate over chunks
        let results = Array1::from_iter(
            (0..n_amps)
                .into_par_iter()
                .step_by(self.chunk_size)
                // get the amps for each chunk
                .map(|start| -> Array1<f64> {
                    let end = min(start + self.chunk_size, n_amps);
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
                            // it is technically impossible for the bin to be larger
                            // than n_bins - 1, but floating point error may bump
                            // the largest amp to above this in some cases
                            let bin = ((amp / width).floor() as usize).min(n_bins - 1);
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
    #[inline(always)]
    fn get_dataset_max(&self, dataset: &Dataset) -> Result<f64> {
        let n_data = dataset.size();
        Ok((0..n_data)
            .into_par_iter()
            .step_by(self.chunk_size)
            .map(|start| -> f64 {
                let end = min(start + self.chunk_size, n_data);
                let array_slice = s![start..end];
                let array: Array1<f64> = dataset
                    .read_slice_1d(array_slice)
                    .expect("Failed to read data.");
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
    use crate::test_utils::MockData;

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
    /// This test uses a fixed maximum.
    #[test]
    fn test_make_amp_histogram_fixed_max() {
        let data = MockData::new().unwrap();
        let amps = Array1::from_vec(vec![0.8, 1.5, 1.2, 3.3, 2., 3.8, 6.1]);
        data.add_dataset("pulse_height", amps).unwrap();
        let nexus_data = data.create(64, 5).unwrap();

        let result = nexus_data.get_amp_histogram(Some(10.), 10);
        assert!(result.is_ok());
        let (hist, max) = result.unwrap();
        assert_eq!(max, 10.);
        assert_eq!(hist, Array1::from_vec(vec![1, 2, 1, 2, 0, 0, 1, 0, 0, 0]))
    }

    /// Test that an amplitude histogram is successfully created for some data.
    /// This test uses a calculated maximum.
    #[test]
    fn test_make_amp_histogram_calculate_max() {
        let data = MockData::new().unwrap();
        let amps = Array1::from_vec(vec![0.8, 1.5, 1.2, 3.3, 2., 3.8, 6.1]);
        data.add_dataset("pulse_height", amps).unwrap();
        let nexus_data = data.create(64, 5).unwrap();

        let result = nexus_data.get_amp_histogram(None, 10);
        assert!(result.is_ok());
        let (hist, max) = result.unwrap();
        assert_eq!(max, 6.1);
        // 10 bins from 0 to 6.1, which means the left bin edges are
        // [0, 0.61, 1.22, 1.83, 2.44, 3.05, 3.66, 4.27, 4.88, 5.49]
        assert_eq!(hist, Array1::from_vec(vec![0, 2, 1, 1, 0, 1, 1, 0, 0, 1]))
    }

    /// Test that `__repr__` includes the filename.
    #[test]
    fn test_repr_includes_filename() {
        let data = test_data();
        let repr = data.__repr__();
        assert_eq!(
            repr,
            "Dataset: ./tests/test_data/HIFI00195790.nxs".to_string()
        );
    }
}
