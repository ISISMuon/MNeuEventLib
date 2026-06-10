use std::path::Path;

use hdf5::{Dataset, Error, File};
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::{pyclass, pymethods};
use pyo3::prelude::{Bound, PyResult};

/// Class for storing a Nexus event file.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Data {
    pub specs: Dataset,
    pub times: Dataset,
    pub amps: Dataset,
    pub frames: Dataset,
    pub frame_times: Dataset,
    pub periods: Option<Dataset>,
    pub n_events: usize,   // the total number of events
    pub n_spec: usize,     // the number of detectors
    pub chunk_size: usize, // the size of the data chunks
}

#[pymethods]
impl Data {
    #[new]
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    fn new(filename: String, n_spec: usize, chunk_size: usize) -> Self {
        let path = Path::new(&filename);
        load_data(path, n_spec, chunk_size).expect("Failed to load data!")
    }

    /// used for testing
    fn get_frame_times<'py>(slf: &Bound<'py, Data>) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let py = slf.py();
        Ok(slf.borrow().frame_times.read_1d().unwrap().to_pyarray(py))
    }
}

pub fn load_data(filename: &Path, n_spec: usize, chunk_size: usize) -> Result<Data, Error> {
    let file = File::open(filename)?;
    let data = file.group("raw_data_1")?.group("detector_1_events")?;

    let specs = data.dataset("event_id")?;
    let times = data.dataset("event_time_offset")?;
    let amps = data.dataset("pulse_height")?;

    let frames = data.dataset("event_index")?;
    let frame_times = data.dataset("event_time_zero")?;
    let periods = data.dataset("period_number").ok();

    let n_events = specs.size();

    Ok(Data {
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
