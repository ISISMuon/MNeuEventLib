use std::path::Path;

use hdf5::{Dataset, Error, File};
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, Bound, PyResult};
use pyo3::exceptions::PyIOError;

/// Class for storing a Nexus event file.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Data {
    pub file: String,
    pub specs: Dataset,
    pub times: Dataset,
    pub amps: Dataset,
    pub frames: Dataset,
    pub frame_times: Dataset,
    pub periods: Dataset,
    pub n_events: usize,   // the total number of events
    pub n_spec: usize,     // the number of detectors
    pub chunk_size: usize, // the size of the data chunks
}

#[pymethods]
impl Data {
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    fn new(filename: String, n_spec: usize, chunk_size: usize) -> PyResult<Self> {
        let path = Path::new(&filename);
        load_data(path, n_spec, chunk_size)
            .map_err(|_| {PyIOError::new_err(format!("Failed to read file {}", filename))})
    }

    /// used for testing
    fn get_frame_times<'py>(slf: &Bound<'py, Data>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let py = slf.py();
        Ok(slf.borrow().frame_times.read_1d().unwrap().to_pyarray(py))
    }
}

/// Load the data in a Nexus event data file into a Data object.
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
pub fn load_data(filename: &Path, n_spec: usize, chunk_size: usize) -> Result<Data, Error> {
    let file = File::open(filename)?;
    let data = file.group("raw_data_1")?.group("detector_1_events")?;

    let specs = data.dataset("event_id")?;
    let times = data.dataset("event_time_offset")?;
    let amps = data.dataset("pulse_height")?;

    let frames = data.dataset("event_index")?;
    let frame_times = data.dataset("event_time_zero")?;
    let periods = data.dataset("period_number")?;

    let n_events = specs.size();

    Ok(Data {
        file: filename.to_str().unwrap().to_string(),
        specs,
        times,
        amps,
        frames,
        frame_times,
        periods,
        n_events,
        n_spec,
        chunk_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the program creates data when you load an existing file.
    #[test]
    fn test_file_load() {
        let path = Path::new("./tests/test_data/HIFI00195790.nxs");
        let data = load_data(path, 960, 1048576);
        assert!(data.is_ok())
    }

    /// Test the program returns an error when the file doesn't exist.
    #[test]
    fn test_file_not_found() {
        let path = Path::new("./mysterious/unreal/fake_data.nxs");
        let data = load_data(path, 960, 1048576);
        assert!(data.is_err())
    }
}
