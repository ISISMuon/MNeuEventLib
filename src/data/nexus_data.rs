use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Mutex;

use anyhow::{Error, Result};
use hdf5::{Dataset, File, Group};
use ndarray::{s, Array1, Array2};
use numpy::{PyArray1, PyArray2, ToPyArray};
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
    #[pyo3(get)]
    pub sample_log_names: Vec<String>,
    pub n_events: usize, // the total number of events
    pub n_frames: usize,
    #[pyo3(get)]
    pub n_spec: usize, // the number of detectors
    pub chunk_size: usize, // the size of the data chunks
}

#[pymethods]
impl NexusData {
    /// used for testing
    fn get_frame_times<'py>(slf: &Bound<'py, NexusData>) -> Bound<'py, PyArray1<u64>> {
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
                    // String logs return `value` as a list[str] rather than an ndarray
                    SampleLog::Str(log) => {
                        output.set_item("name".to_string(), log.name)?;
                        output.set_item("time".to_string(), log.time.to_pyarray(py))?;
                        output.set_item(
                            "value".to_string(),
                            log.value
                                .iter()
                                .map(|v| v.to_string())
                                .collect::<Vec<String>>(),
                        )?;
                        output.set_item("unit".to_string(), log.unit)?;
                    }
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

    /// Get a histogram of the amplitudes and the maximum height summed over all detectors.
    ///
    /// Parameters
    /// ----------
    /// max_height: float, optional
    ///     The maximum height to use. If not provided, the maximum
    ///     amplitude in the data is used.
    /// n_bins: int, default 10
    ///     The number of equal-width bins to use.
    /// detector: int, optional
    ///     The detector to create an amplitude histogram for.
    ///     If not provided, all detectors are used.
    #[pyo3(name = "get_amp_histogram", signature = (max_height=None, n_bins=10, detector=None))]
    fn get_amp_histogram_py<'py>(
        &self,
        py: Python<'py>,
        max_height: Option<f64>,
        n_bins: usize,
        detector: Option<usize>,
    ) -> Result<(Bound<'py, PyArray1<usize>>, f64)> {
        let (results, max) = self.get_amp_histogram(max_height, n_bins, detector)?;
        Ok((results.to_pyarray(py), max))
    }

    /// Get a histogram of the amplitudes for each of several detectors,
    /// and the maximum height.
    ///
    /// Every detector is histogrammed over the same bin edges, so the
    /// results can be compared directly.
    ///
    /// Parameters
    /// ----------
    /// detectors: Sequence[int]
    ///     The detectors to create amplitude histograms for.
    ///     Must not be empty or contain duplicates.
    /// max_height: float, optional
    ///     The maximum height to use. If not provided, the maximum
    ///     amplitude in the data is used.
    /// n_bins: int, default 10
    ///     The number of equal-width bins to use.
    ///
    /// Returns
    /// -------
    /// np.ndarray
    ///     An array of shape (len(detectors), n_bins), where row i is the
    ///     histogram for detectors[i].
    /// float
    ///     The maximum height used.
    #[pyo3(name = "get_amp_histograms", signature = (detectors, max_height=None, n_bins=10))]
    fn get_amp_histograms_py<'py>(
        &self,
        py: Python<'py>,
        detectors: Vec<usize>,
        max_height: Option<f64>,
        n_bins: usize,
    ) -> Result<(Bound<'py, PyArray2<usize>>, f64)> {
        let (results, max) = self.get_amp_histograms(max_height, n_bins, &detectors)?;
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
    ///
    /// If `detector` is given, only events recorded by that detector are counted.
    #[inline(always)]
    fn get_amp_histogram(
        &self,
        max_height: Option<f64>,
        n_bins: usize,
        detector: Option<usize>,
    ) -> Result<(Array1<usize>, f64)> {
        // a single detector is just the one-row case of get_amp_histograms
        if let Some(detector) = detector {
            let (hists, max) = self.get_amp_histograms(max_height, n_bins, &[detector])?;
            return Ok((hists.row(0).to_owned(), max));
        }

        // with no detector given we count every event, so unlike get_amp_histograms
        // we never need to read the spectrum numbers
        let max = self.resolve_max_height(max_height)?;
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
                    self.amps
                        .read_slice_1d(s![start..end])
                        .expect("Failed to read amplitude data.")
                })
                // bin amps for each chunk
                .map(|amps| {
                    let mut array = Array1::zeros(n_bins);
                    for amp in amps {
                        array[amp_bin(amp, max, width, n_bins)] += 1
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

    /// Get a histogram of amplitudes for each of several detectors, and the max height.
    ///
    /// Row `i` of the result is the histogram for `detectors[i]`. Every detector is
    /// binned over the same edges in a single pass over the file, so the rows can be
    /// compared directly and the cost does not grow with the number of detectors.
    fn get_amp_histograms(
        &self,
        max_height: Option<f64>,
        n_bins: usize,
        detectors: &[usize],
    ) -> Result<(Array2<usize>, f64)> {
        if detectors.is_empty() {
            return Err(Error::msg(
                "Attempted to get amplitude histograms, but no detectors were given.",
            ));
        }
        // map each detector onto its row of the output, so that binning an event is a
        // lookup rather than a search. this also catches duplicates, which would
        // otherwise silently give a detector two rows of the result
        let mut rows: Vec<Option<usize>> = vec![None; self.n_spec];
        for (row, &detector) in detectors.iter().enumerate() {
            if detector >= self.n_spec {
                return Err(Error::msg(format!(
                    "Attempted to get amplitude histogram for detector {detector}, but instrument only has {} detectors (numbered 0 to {}).",
                    self.n_spec,
                    self.n_spec - 1
                )));
            }
            if rows[detector].is_some() {
                return Err(Error::msg(format!(
                    "Attempted to get amplitude histograms, but detector {detector} was given more than once."
                )));
            }
            rows[detector] = Some(row);
        }

        let max = self.resolve_max_height(max_height)?;
        if n_bins == 0 {
            return Err(Error::msg("n_bins must be greater than 0."));
        }
        let width = max / n_bins as f64;
        let n_amps = self.amps.size();
        let n_detectors = detectors.len();

        // accumulate one set of histograms per worker, then combine them at the end.
        // current_thread_index may be None (if it runs on a thread that isn't a Rayon
        // thread somehow) so we keep a fallback slot just in case
        let n_threads = rayon::current_num_threads() + 1;
        let fallback_slot = n_threads - 1;
        let accumulators: Vec<Mutex<Option<Array2<usize>>>> =
            (0..n_threads).map(|_| Mutex::new(None)).collect();

        (0..n_amps)
            .into_par_iter()
            .step_by(self.chunk_size)
            .for_each(|start| {
                let end = min(start + self.chunk_size, n_amps);
                let array_slice = s![start..end];
                let amps: Array1<f64> = self
                    .amps
                    .read_slice_1d(array_slice)
                    .expect("Failed to read amplitude data.");
                let specs: Array1<u32> = self
                    .specs
                    .read_slice_1d(array_slice)
                    .expect("Failed to read spectrum data.");

                let thread = rayon::current_thread_index().unwrap_or(fallback_slot);
                let mut acc_mutex = accumulators[thread]
                    .lock()
                    .expect("accumulator thread panicked!");
                let acc = acc_mutex.get_or_insert_with(|| Array2::zeros((n_detectors, n_bins)));

                for (i, amp) in amps.into_iter().enumerate() {
                    // skip events recorded by a detector we weren't asked about
                    let Some(row) = rows.get(specs[i] as usize).copied().flatten() else {
                        continue;
                    };
                    acc[[row, amp_bin(amp, max, width, n_bins)]] += 1
                }
            });

        // combine the per-worker histograms
        let results = accumulators
            .into_iter()
            .filter_map(|thread| thread.into_inner().expect("accumulator thread panicked!"))
            .reduce(|mut acc, other| {
                acc += &other;
                acc
            })
            // every worker may have been left unused if the file holds no events
            .unwrap_or_else(|| Array2::zeros((n_detectors, n_bins)));

        Ok((results, max))
    }

    /// Work out the maximum height to use for an amplitude histogram.
    #[inline(always)]
    fn resolve_max_height(&self, max_height: Option<f64>) -> Result<f64> {
        match max_height {
            Some(height) if height.is_finite() => Ok(height),
            Some(_) => Err(Error::msg("max_height must be finite.")),
            None => self.get_dataset_max(&self.amps),
        }
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

/// Work out which bin of an amplitude histogram an amplitude falls in.
#[inline(always)]
fn amp_bin(amp: f64, max: f64, width: f64, n_bins: usize) -> usize {
    // anything over the max is put in the last bin
    if amp >= max {
        n_bins - 1
    } else {
        // it is technically impossible for the bin to be larger
        // than n_bins - 1, but floating point error may bump
        // the largest amp to above this in some cases
        ((amp / width).floor() as usize).min(n_bins - 1)
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
    use std::str::FromStr;

    use super::*;
    use crate::test_utils::MockData;
    use hdf5::types::{VarLenAscii, VarLenUnicode};

    fn test_data() -> (NexusData, std::sync::MutexGuard<'static, ()>) {
        let guard = crate::test_utils::lock_hdf5_test();
        let path = Path::new("./tests/test_data/HIFI00195790.nxs");
        (load_data(path, 64, 1048576).unwrap(), guard)
    }

    /// Test the program creates data when you load an existing file.
    #[test]
    fn test_file_load() {
        let _guard = crate::test_utils::lock_hdf5_test();
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
        let (data, _guard) = test_data();

        assert_eq!(data.sample_log_names, vec!("Temp".to_string()))
    }

    /// Test that an existing sample log can be read successfully.
    #[test]
    fn test_load_sample_log() {
        let (data, _guard) = test_data();

        let log = data.get_sample_log(&"Temp".to_string());
        assert!(log.is_ok());

        let value_log = log.unwrap();
        assert!(matches!(value_log, SampleLog::F32(_)))
    }

    /// Test that a non-real sample log throws an error.
    #[test]
    fn test_load_sample_log_not_found() {
        let (data, _guard) = test_data();

        let log = data.get_sample_log(&"Lunch".to_string());
        assert!(log.is_err())
    }

    /// Test a variable-length UTF-8 sample log is read.
    #[test]
    fn test_load_string_sample_log() {
        let mock = MockData::new().unwrap();
        mock.add_sample_log(
            "status",
            Array1::from_vec(vec![0., 3., 10.]),
            Array1::from_vec(
                ["IDLE", "RUNNING", "PAUSED"]
                    .iter()
                    .map(|s| VarLenUnicode::from_str(s).unwrap())
                    .collect::<Vec<VarLenUnicode>>(),
            ),
        )
        .unwrap();
        let data = mock.create(64, 1048576).unwrap();

        let log = data.get_sample_log(&"status".to_string()).unwrap();

        match log {
            SampleLog::Str(log) => {
                let values: Vec<String> = log.value.iter().map(|v| v.to_string()).collect();
                assert_eq!(values, vec!["IDLE", "RUNNING", "PAUSED"]);
                assert_eq!(log.time, Array1::from_vec(vec![0., 3., 10.]));
            }
            _ => panic!("a string sample log should load as SampleLog::Str"),
        }
    }

    /// Test an ASCII string log is normalised onto the same Str variant
    #[test]
    fn test_load_ascii_sample_log_normalises_to_unicode() {
        let mock = MockData::new().unwrap();
        mock.add_sample_log(
            "status",
            Array1::from_vec(vec![0., 1.]),
            Array1::from_vec(
                ["IDLE", "RUNNING"]
                    .iter()
                    .map(|s| VarLenAscii::from_ascii(s).unwrap())
                    .collect::<Vec<VarLenAscii>>(),
            ),
        )
        .unwrap();
        let data = mock.create(64, 1048576).unwrap();

        let log = data.get_sample_log(&"status".to_string()).unwrap();

        match log {
            SampleLog::Str(log) => {
                let values: Vec<String> = log.value.iter().map(|v| v.to_string()).collect();
                assert_eq!(values, vec!["IDLE", "RUNNING"]);
            }
            _ => panic!("an ASCII sample log should load as SampleLog::Str"),
        }
    }

    /// Test that an amplitude histogram is successfully created for some data.
    /// This test uses a fixed maximum.
    #[test]
    fn test_make_amp_histogram_fixed_max() {
        let data = MockData::new().unwrap();
        let amps = Array1::from_vec(vec![0.8, 1.5, 1.2, 3.3, 2., 3.8, 6.1]);
        data.add_dataset("pulse_height", amps).unwrap();
        let nexus_data = data.create(64, 5).unwrap();

        let result = nexus_data.get_amp_histogram(Some(10.), 10, None);
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

        let result = nexus_data.get_amp_histogram(None, 10, None);
        assert!(result.is_ok());
        let (hist, max) = result.unwrap();
        assert_eq!(max, 6.1);
        // 10 bins from 0 to 6.1, which means the left bin edges are
        // [0, 0.61, 1.22, 1.83, 2.44, 3.05, 3.66, 4.27, 4.88, 5.49]
        assert_eq!(hist, Array1::from_vec(vec![0, 2, 1, 1, 0, 1, 1, 0, 0, 1]))
    }

    /// Test that an amplitude histogram can be made for a single detector,
    /// counting only the events recorded by that detector.
    #[test]
    fn test_make_amp_histogram_single_detector() {
        let data = MockData::new().unwrap();
        let amps = Array1::from_vec(vec![0.8, 1.5, 1.2, 3.3, 2., 3.8, 6.1]);
        let specs: Array1<u32> = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0]);
        data.add_dataset("pulse_height", amps).unwrap();
        data.add_dataset("event_id", specs).unwrap();
        let nexus_data = data.create(64, 5).unwrap();

        // detector 1 recorded the amplitudes 1.5, 3.3 and 3.8
        let (hist, max) = nexus_data
            .get_amp_histogram(Some(10.), 10, Some(1))
            .unwrap();
        assert_eq!(max, 10.);
        assert_eq!(hist, Array1::from_vec(vec![0, 1, 0, 2, 0, 0, 0, 0, 0, 0]));

        // detector 0 recorded the amplitudes 0.8, 1.2, 2. and 6.1
        let (hist, _) = nexus_data
            .get_amp_histogram(Some(10.), 10, Some(0))
            .unwrap();
        assert_eq!(hist, Array1::from_vec(vec![1, 1, 1, 0, 0, 0, 1, 0, 0, 0]))
    }

    /// Test a detector that recorded no events gives an empty histogram
    /// rather than an error.
    #[test]
    fn test_make_amp_histogram_detector_no_events() {
        let data = MockData::new().unwrap();
        let amps = Array1::from_vec(vec![0.8, 1.5, 1.2]);
        let specs: Array1<u32> = Array1::from_vec(vec![0, 1, 0]);
        data.add_dataset("pulse_height", amps).unwrap();
        data.add_dataset("event_id", specs).unwrap();
        let nexus_data = data.create(64, 5).unwrap();

        let (hist, _) = nexus_data
            .get_amp_histogram(Some(10.), 10, Some(2))
            .unwrap();
        assert_eq!(hist, Array1::<usize>::zeros(10))
    }

    /// Test that histogramming several detectors in one pass gives the same
    /// result as histogramming each of them separately, and that the rows
    /// come back in the order the detectors were asked for.
    #[test]
    fn test_make_amp_histograms_multiple_detectors() {
        let data = MockData::new().unwrap();
        let amps = Array1::from_vec(vec![0.8, 1.5, 1.2, 3.3, 2., 3.8, 6.1]);
        let specs: Array1<u32> = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0]);
        data.add_dataset("pulse_height", amps).unwrap();
        data.add_dataset("event_id", specs).unwrap();
        let nexus_data = data.create(64, 5).unwrap();

        // ask for detector 1 first, so that the rows are not simply in detector order
        let (hists, max) = nexus_data
            .get_amp_histograms(Some(10.), 10, &[1, 0])
            .unwrap();
        assert_eq!(max, 10.);
        assert_eq!(hists.shape(), [2, 10]);

        for (row, detector) in [1, 0].iter().enumerate() {
            let (expected, _) = nexus_data
                .get_amp_histogram(Some(10.), 10, Some(*detector))
                .unwrap();
            assert_eq!(hists.row(row).to_owned(), expected);
        }
    }

    /// Test that asking for no detectors is an error, rather than an
    /// empty result that would plot as a blank figure.
    #[test]
    fn test_make_amp_histograms_no_detectors_errors() {
        let (data, _guard) = test_data();

        let result = data.get_amp_histograms(Some(10.), 10, &[]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "Attempted to get amplitude histograms, but no detectors were given.".to_string()
        )
    }

    /// Test that asking for the same detector twice is an error.
    #[test]
    fn test_make_amp_histograms_duplicate_detector_errors() {
        let (data, _guard) = test_data();

        let result = data.get_amp_histograms(Some(10.), 10, &[0, 1, 0]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "Attempted to get amplitude histograms, but detector 0 was given more than once."
                .to_string()
        )
    }

    /// Test that a detector outside the instrument is an error, and that the
    /// message says which detector numbers are valid.
    #[test]
    fn test_make_amp_histograms_detector_out_of_range_errors() {
        let (data, _guard) = test_data();

        let result = data.get_amp_histograms(Some(10.), 10, &[0, 64]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "Attempted to get amplitude histogram for detector 64, but instrument only has 64 detectors (numbered 0 to 63).".to_string()
        )
    }

    /// Test that `__repr__` includes the filename.
    #[test]
    fn test_repr_includes_filename() {
        let (data, _guard) = test_data();
        let repr = data.__repr__();
        assert_eq!(
            repr,
            "Dataset: ./tests/test_data/HIFI00195790.nxs".to_string()
        );
    }
}
