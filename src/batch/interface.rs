use crate::data::save::sanitise::nexus_data::{get_period_info, save_default};
use crate::data::{NexusData, SaveFile, WiMDAFile};
use crate::filters::Filters;
use crate::stats::Histogram;
use anyhow::{Error, Result};
use ndarray::Array1;
use numpy::{PyArray3, ToPyArray};
use pyo3::prelude::{pyclass, pymethods, Borrowed, Bound, FromPyObject, PyAny};
use pyo3::types::{PyInt, PyString};
use std::path::PathBuf;

pub type PyHist<'py> = Bound<'py, PyArray3<i32>>;

/// The reserved word selecting every filter set.
const ALL_KEYWORD: &str = "all";

/// The index of a filter set (and its corresponding result) within a
/// [`BatchData`], or a request to apply an operation to every filter set.
///
/// From Python this can be constructed from an integer (e.g. `0`, `1`, ...),
/// the string `"all"` (case-insensitive), or the label of a filter set.
pub enum FilterIndex {
    /// Apply the operation to every filter set.
    All,
    /// Apply the operation to a single filter set at this index.
    Index(usize),
    /// Apply the operation to the single filter set carrying this label.
    Label(String),
}

impl<'a, 'py> FromPyObject<'a, 'py> for FilterIndex {
    type Error = Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self> {
        // If index is given as an integer, turn into Index integer
        if let Ok(index) = obj.cast::<PyInt>() {
            return Ok(FilterIndex::Index(index.extract()?));
        // If index is given as a string, it is either 'all' or a label.
        // We cannot tell a valid label from a typo here, as we have no access
        // to the BatchData; the lookup in `resolve_indices` raises instead.
        } else if let Ok(string) = obj.cast::<PyString>() {
            let string = string.extract::<String>()?;
            if string.to_lowercase() == ALL_KEYWORD {
                return Ok(FilterIndex::All);
            }
            return Ok(FilterIndex::Label(string));
        }
        // If index is anything else, fail
        Err(Error::msg(
            "Filter index must be a number, a label, or 'all'",
        ))
    }
}

/// An interface for processing multiple batches of data.
///
/// Each filter set `i` has its own corresponding result `i`; when
/// `calculate` is run, `results[i]` is calculated from `dataset` using
/// `filters[i]`.
///
/// Filter-mutating methods take an extra `index: FilterIndex` parameter,
/// which is either `"all"` (apply the change to every filter
/// set), `i` (apply the change to filter set `i` only), or the label of
/// a filter set (apply the change to the set carrying that label).
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct BatchData {
    #[pyo3(get)]
    pub dataset: Option<NexusData>,
    pub results: Vec<Histogram>,
    pub filters: Vec<Filters>,
    labels: Vec<Option<String>>, // the optional label of each filter set
    data_changed: Vec<bool>,     // whether data has changed since last calculation, per filter set
}

#[pymethods]
impl BatchData {
    /// Create a new BatchData object.
    ///
    /// Parameters
    /// ----------
    /// filename: str
    ///     The filename of the NeXuS file to load.
    /// n_spec: int
    ///     The number of detectors (spectra) in the dataset.
    /// n_filter_sets: int
    ///     The number of filter sets (and corresponding results) to create.
    /// chunk_size: int
    ///     The chunk size to use when reading the dataset.
    #[new]
    #[pyo3(signature = (filename, n_spec, n_filter_sets, chunk_size=1048576))]
    pub fn new(
        filename: String,
        n_spec: usize,
        n_filter_sets: usize,
        chunk_size: usize,
    ) -> Result<Self> {
        if n_filter_sets == 0 {
            return Err(Error::msg("n_filter_sets must be greater than 0."));
        }
        let dataset = NexusData::new(filename, n_spec, chunk_size)?;
        Ok(BatchData {
            dataset: Some(dataset),
            results: (0..n_filter_sets)
                .map(|_| Histogram::new(0, 32768, 2048))
                .collect(),
            filters: (0..n_filter_sets).map(|_| Filters::new()).collect(),
            labels: vec![None; n_filter_sets],
            data_changed: vec![true; n_filter_sets],
        })
    }

    /// Set the dataset for the current Data object.
    ///
    /// Parameters
    /// ----------
    /// filename: String
    ///     The name of the dataset.
    #[pyo3(signature = (filename, n_spec, chunk_size=1048576))]
    pub fn set_data(&mut self, filename: String, n_spec: usize, chunk_size: usize) -> Result<()> {
        let dataset = NexusData::new(filename, n_spec, chunk_size)?;
        self.dataset = Some(dataset);
        Ok(())
    }

    /// Calculate the histograms for the current data and each filter set.
    ///
    /// Returns
    /// -------
    /// BatchData
    ///     This object, with `results[i]` holding the histogram calculated
    ///     from `dataset` and `filters[i]`, for each `i`.
    pub fn calculate(&mut self) -> Result<BatchData> {
        match &self.dataset {
            Some(dataset) => {
                for i in 0..self.n_batches() {
                    if self.data_changed[i] {
                        let result = self.results[i].calculate(dataset, &self.filters[i])?;
                        self.data_changed[i] = false;
                        self.results[i] = result;
                    }
                }
                Ok(self.clone())
            }
            None => Err(Error::msg(
                "Dataset has not been set! Set with the set_data() method.",
            )),
        }
    }

    /// Force histograms to be recalculated even if the data hasn't changed.
    pub fn invalidate_cache(&mut self) {
        self.data_changed = vec![true; self.n_batches()]
    }

    /// Set histogram settings for one or all filter sets.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// min_time: float
    ///     The minimum time bound for the histogram.
    /// max_time: float
    ///     The maximum time bound for the histogram.
    /// n_bins: int
    ///     The number of bins to divide the time range into.
    pub fn set_histogram_settings(
        &mut self,
        index: FilterIndex,
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
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.results[i] =
                Histogram::new((min_time * 1e3) as u32, (max_time * 1e3) as u32, n_bins);
        }
        Ok(())
    }

    /// Set the type for the time filters.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// filter_type: str
    ///     The type for the time filters. Must be 'exclude' or 'include'.
    pub fn set_time_type(&mut self, index: FilterIndex, filter_type: String) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].set_time_type(filter_type.clone())?;
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Add a time filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// name: str
    ///     The name of the time filter. Must be unique within each modified filter set.
    /// start: float
    ///     The start point for the time filter.
    /// end: float
    ///     The end point for the time filter.
    pub fn add_time_filter(
        &mut self,
        index: FilterIndex,
        name: String,
        start: f64,
        end: f64,
    ) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].add_time_filter(name.clone(), start, end)?;
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Remove a time filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// name: str
    ///     The name of the time filter to remove.
    pub fn remove_time_filter(&mut self, index: FilterIndex, name: String) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].remove_time_filter(name.clone())?;
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Add a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// name: str
    ///     The name of the log filter. Must be unique within each modified filter set.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// lower: float
    ///     The lower bound for the log filter.
    /// upper: float
    ///     The upper bound for the log filter.
    pub fn add_log_filter(
        &mut self,
        index: FilterIndex,
        name: String,
        log: String,
        lower: f64,
        upper: f64,
    ) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].add_log_filter(name.clone(), log.clone(), Some(lower), Some(upper))?;
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Remove a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// name: str
    ///     The name of the log filter to remove.
    pub fn remove_log_filter(&mut self, index: FilterIndex, name: String) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].remove_log_filter(name.clone())?;
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Add a sample log filter for all data above a certain value.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// name: str
    ///     The name of the log filter. Must be unique within each modified filter set.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// lower: float
    ///     The lower bound for the log filter.
    pub fn add_log_filter_above(
        &mut self,
        index: FilterIndex,
        name: String,
        log: String,
        lower: f64,
    ) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].add_log_filter_above(name.clone(), log.clone(), lower)?;
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Add a sample log filter for all data below a certain value.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// name: str
    ///     The name of the log filter. Must be unique within each modified filter set.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// upper: float
    ///     The upper bound for the log filter.
    pub fn add_log_filter_below(
        &mut self,
        index: FilterIndex,
        name: String,
        log: String,
        upper: f64,
    ) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].add_log_filter_below(name.clone(), log.clone(), upper)?;
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Set the amplitude filter for a detector.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// detector: int
    ///     The detector to set a filter for.
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    pub fn set_amp(&mut self, index: FilterIndex, detector: usize, amp: f64) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].set_amp(detector, amp);
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Set an amplitude filter for all detectors that don't have one defined.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index or label of the filter set to modify.
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    pub fn set_amps_baseline(&mut self, index: FilterIndex, amp: f64) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.filters[i].set_amps_baseline(amp);
            self.data_changed[i] = true;
        }
        Ok(())
    }

    /// Label a filter set, so it can be addressed by that label instead of
    /// by its index.
    ///
    /// Note that unlike the filter-mutating methods, `index` here must be an
    /// integer: a label cannot be given to every filter set at once, as
    /// labels are unique.
    ///
    /// Parameters
    /// ----------
    /// index: int
    ///     The index of the filter set to label.
    /// label: str | None
    ///     The label to give the filter set. Must not be empty, must not be
    ///     'all', and must not already be in use by another filter set.
    ///     Pass `None` to remove an existing label.
    pub fn set_label(&mut self, index: usize, label: Option<String>) -> Result<()> {
        self.check_index(index)?;
        match label {
            None => self.labels[index] = None,
            Some(label) => {
                if label.is_empty() {
                    return Err(Error::msg("A filter set label cannot be empty."));
                }
                if label.to_lowercase() == ALL_KEYWORD {
                    return Err(Error::msg(format!(
                        "'{label}' cannot be used as a filter set label, \
                         as '{ALL_KEYWORD}' selects every filter set."
                    )));
                }
                // relabelling a set with its own label is a no-op, not a clash
                if let Some(existing) = self.find_label(&label) {
                    if existing != index {
                        return Err(Error::msg(format!(
                            "Filter set {existing} is already labelled '{label}'. \
                             Labels must be unique."
                        )));
                    }
                }
                self.labels[index] = Some(label);
            }
        }
        Ok(())
    }

    /// The label of each filter set, in index order, with `None` for the
    /// filter sets that are unlabelled.
    #[getter]
    pub fn labels(&self) -> Vec<Option<String>> {
        self.labels.clone()
    }

    /// Add two BatchData objects together componentwise.
    ///
    /// Note that labels are dropped by this function, as each resulting
    /// filter set is built from a filter set of *each* object and so
    /// corresponds to neither.
    ///
    /// Parameters
    /// ----------
    /// other: BatchData
    ///     The data to add to this dataset.
    pub fn add(&self, other: BatchData) -> Result<BatchData> {
        if self.n_batches() != other.n_batches() {
            return Err(Error::msg(
                "Can only add data objects with the same number of batches.",
            ));
        }

        let dataset = self.combine_data(&other.dataset)?;

        let results = self.results.clone();

        let data_changed = vec![true; results.len()];

        let mut filters = self.filters.clone();
        for (i, filter) in filters.iter_mut().enumerate() {
            filter.extend(other.filters[i].clone())
        }

        Ok(BatchData {
            dataset,
            filters,
            labels: vec![None; results.len()],
            results,
            data_changed,
        })
    }

    /// Let `+` (add) add objects.
    pub fn __add__(&self, other: BatchData) -> Result<BatchData> {
        self.add(other)
    }

    /// Concatenate BatchData objects.
    ///
    /// Labels are carried over, as each filter set of each object becomes a
    /// filter set of the result unchanged. It is therefore an error for both
    /// objects to use the same label.
    ///
    /// Parameters
    /// ----------
    /// other: BatchData
    ///     The data to concatenate with this dataset.
    pub fn concatenate(&self, other: BatchData) -> Result<BatchData> {
        let dataset = self.combine_data(&other.dataset)?;

        let mut filters = self.filters.clone();
        filters.extend(other.filters);

        let mut results = self.results.clone();
        results.extend(other.results);

        // labels stay aligned with their filter sets automatically, but the
        // two objects were labelled independently, so may clash
        if let Some(clash) = other
            .labels
            .iter()
            .flatten()
            .find(|label| self.find_label(label).is_some())
        {
            return Err(Error::msg(format!(
                "Both objects have a filter set labelled '{clash}'. \
                 Labels must be unique; relabel one with set_label()."
            )));
        }
        let mut labels = self.labels.clone();
        labels.extend(other.labels);

        let data_changed = vec![true; filters.len()];

        Ok(BatchData {
            dataset,
            filters,
            labels,
            results,
            data_changed,
        })
    }

    /// Let `&` (and) concatenate objects.
    pub fn __and__(&self, other: BatchData) -> Result<BatchData> {
        self.concatenate(other)
    }

    /// Take all combinations of filters in two BatchData objects;
    /// i.e. this computes the cartesian product.
    ///
    /// Note that histogram settings are reset by this function, and labels
    /// are dropped, as each resulting filter set is built from a *pair* of
    /// filter sets and so corresponds to neither of them.
    ///
    /// Parameters
    /// ----------
    /// other: BatchData
    ///     The data to combine with this dataset.
    pub fn combinations(&self, other: BatchData) -> Result<BatchData> {
        let dataset = self.combine_data(&other.dataset)?;

        let n = self.n_batches();
        let m = other.n_batches();
        let result_size = n * m;

        let results = vec![Histogram::new(0, 32768, 2048); result_size];

        let data_changed = vec![true; result_size];

        let mut filters = Vec::<Filters>::with_capacity(result_size);
        for i in 0..n {
            for j in 0..m {
                // pushing in this order puts the pair (i, j) at index i * m + j
                let mut combined = self.filters[i].clone();
                combined.extend(other.filters[j].clone());
                filters.push(combined);
            }
        }

        Ok(BatchData {
            dataset,
            filters,
            labels: vec![None; result_size],
            results,
            data_changed,
        })
    }

    /// Let `*` (multiply) combine objects.
    pub fn __mul__(&self, other: BatchData) -> Result<BatchData> {
        self.combinations(other)
    }

    /// Save a filter set's result to a file.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     The index or label of the filter set/result to save. If 'all',
    ///     an index number will be appended to each filename.
    /// filename: str
    ///     The filename for the saved file.
    /// default: bool
    ///     Whether to use default values for the missing meta-data (this is
    ///     needed because the event data files has mistakes/problems).
    ///     This allows the file to be read by Mantid even if the event file
    ///     is incomplete.
    /// autofill: bool
    ///     Whether to use automatically fill the file with
    ///     default values for the missing meta-data (this is
    ///     needed because the event data files have mistakes/problems).
    ///     This allows the file to be read by Mantid even if the event file
    ///     is incomplete.
    /// ref_file: str
    ///     The reference file for the saved file. (must be a Nexus file)
    ///     Contains "correct" data that should be copied to the output file.
    ///     This is only need it the reference file needed is not the standard
    ///     muon nexus v2 file. The ref_file is generated from tools/make_default.py.
    #[pyo3(signature = (index, filename, autofill=true, ref_file = (PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("files/muon_ref.nxs")).display().to_string()))]
    pub fn save(
        &self,
        index: FilterIndex,
        filename: String,
        autofill: bool,
        ref_file: String,
    ) -> Result<()> {
        let filename_stem = if filename.to_lowercase().ends_with(".nxs") {
            filename.clone()[..(filename.len() - 4)].to_string()
        } else {
            filename.clone()
        };

        // a label names exactly one filter set, so saves as a single index does
        let index = match index {
            FilterIndex::Label(label) => FilterIndex::Index(self.label_to_index(&label)?),
            other => other,
        };

        match index {
            FilterIndex::Index(i) => {
                self.check_index(i)?;
                if self.results[i].hist.shape() == [0, 0, 0] {
                    return Err(Error::msg(
                        "Cannot save as results have not been calculated.",
                    ));
                }
                let dataset = self.dataset.as_ref().unwrap();
                let wimda_file = WiMDAFile::new(dataset, &self.filters[i], &self.results[i])?;
                wimda_file.save_file(format!("{filename_stem}.nxs"), &dataset.file)?;
                if autofill {
                    self.save_nexus(format!("{filename_stem}.nxs"), ref_file.clone())?;
                }
            }
            FilterIndex::All => {
                if self.results.iter().any(|r| r.hist.shape() == [0, 0, 0]) {
                    return Err(Error::msg(
                        "Cannot save as results have not been calculated.",
                    ));
                }
                let dataset = self.dataset.as_ref().unwrap();
                for i in 0..self.n_batches() {
                    let wimda_file = WiMDAFile::new(dataset, &self.filters[i], &self.results[i])?;
                    wimda_file.save_file(format!("{filename_stem}_{i}.nxs"), &dataset.file)?;
                    if autofill {
                        self.save_nexus(format!("{filename_stem}_{i}.nxs"), ref_file.clone())?;
                    }
                }
            }
            // resolved into an Index above
            FilterIndex::Label(_) => unreachable!(),
        }
        Ok(())
    }

    /// Get a calculated histogram.
    ///
    /// Parameters
    /// ----------
    /// index: int
    ///     The index of the histogram to get.
    pub fn get_histogram<'py>(
        slf: &Bound<'py, BatchData>,
        index: FilterIndex,
    ) -> Result<Vec<PyHist<'py>>> {
        let py = slf.py();
        Ok(slf
            .borrow()
            .resolve_indices(&index)?
            .into_iter()
            .map(|i| slf.borrow().results[i].hist.to_pyarray(py))
            .collect())
    }

    /// Get the number of events in each histogram.
    ///
    /// Parameters
    /// ----------
    /// index: int
    ///     The index of the histogram to get.
    pub fn get_n_events(&self, index: FilterIndex) -> Result<Vec<usize>> {
        Ok(self
            .resolve_indices(&index)?
            .into_iter()
            .map(|i| self.results[i].n)
            .collect())
    }

    fn __repr__(&self) -> String {
        let mut string = match &self.dataset {
            Some(data) => data.__repr__(),
            None => "No data set.".to_string(),
        };
        for (i, (filters, results)) in self.filters.iter().zip(self.results.iter()).enumerate() {
            // the index is shown whether or not the set is labelled, as
            // filter sets stay addressable by position either way
            let label = match &self.labels[i] {
                Some(label) => format!(" ({label})"),
                None => String::new(),
            };
            string += &format!(
                "\n\nFilter set {i}{label}:\n{}\n\n{}",
                filters.__repr__(),
                results.__repr__()
            );
        }
        string
    }

    /// The number of filter sets (and results) held by this object.
    pub fn __len__(&self) -> usize {
        self.n_batches()
    }
}

impl BatchData {
    /// Create an empty BatchData object.
    pub fn empty(n: usize) -> BatchData {
        BatchData {
            dataset: None,
            results: vec![Histogram::new(0, 32768, 2048); n],
            filters: vec![Filters::new(); n],
            labels: vec![None; n],
            data_changed: vec![true; n],
        }
    }

    /// Resolve a [`FilterIndex`] into a list of valid filter set indices,
    /// checking bounds along the way.
    fn resolve_indices(&self, index: &FilterIndex) -> Result<Vec<usize>> {
        match index {
            FilterIndex::All => Ok((0..self.n_batches()).collect()),
            FilterIndex::Index(i) => {
                self.check_index(*i)?;
                Ok(vec![*i])
            }
            FilterIndex::Label(label) => Ok(vec![self.label_to_index(label)?]),
        }
    }

    /// Find the filter set carrying a given label, or error if not found.
    fn label_to_index(&self, label: &str) -> Result<usize> {
        self.find_label(label).ok_or_else(|| {
            // PyO3 downcasts back out of anyhow, so this raises a real KeyError
            let available = self.labels.iter().flatten().cloned().collect::<Vec<_>>();
            let available = if available.is_empty() {
                "no filter sets are labelled".to_string()
            } else {
                format!("available labels are {}", available.join(", "))
            };
            Error::msg(format!("No filter set labelled '{label}': {available}."))
        })
    }

    /// Get the index of a given label.
    fn find_label(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l.as_deref() == Some(label))
    }

    /// Check that a given index is valid for this BatchData's filter sets.
    fn check_index(&self, index: usize) -> Result<()> {
        if index >= self.n_batches() {
            return Err(Error::msg(format!(
                "Index {index} out of range: only {} filter sets exist.",
                self.filters.len()
            )));
        }
        Ok(())
    }

    /// Check that two datasets are the same or at least one is None;
    /// if both are None return None, if one is None or both are equal, return the dataset,
    /// if datasets are different, throw an error.
    fn combine_data(&self, other: &Option<NexusData>) -> Result<Option<NexusData>> {
        let data = &self.dataset;
        if let Some(dataset) = data {
            match other {
                // this dataset exists, the other has no data
                None => Ok(data.clone()),
                // other dataset exists, check compatibility
                Some(other_dataset) => {
                    if dataset.filename == other_dataset.filename {
                        Ok(data.clone())
                    } else {
                        Err(Error::msg(
                            "Tried to combine BatchData objects with different data!",
                        ))
                    }
                }
            }
        } else {
            // just take other dataset (which is some data or also None)
            Ok(other.clone())
        }
    }

    /// Turn an array of n+1 elements into n time filters across the batches.
    pub fn array_to_time_filters(&mut self, name: String, input: Array1<f64>) -> Result<()> {
        self.check_array_len(&input)?;
        for i in 0..self.n_batches() {
            self.add_time_filter(FilterIndex::Index(i), name.clone(), input[i], input[i + 1])?
        }
        Ok(())
    }

    /// Turn an array of n+1 elements into n sample log filters across the batches.
    pub fn array_to_log_filters(
        &mut self,
        name: String,
        log: String,
        input: Array1<f64>,
    ) -> Result<()> {
        self.check_array_len(&input)?;
        for i in 0..self.n_batches() {
            self.add_log_filter(
                FilterIndex::Index(i),
                name.clone(),
                log.clone(),
                input[i],
                input[i + 1],
            )?
        }
        Ok(())
    }

    /// Check that an array holds enough elements to bound every batch.
    fn check_array_len(&self, input: &Array1<f64>) -> Result<()> {
        if input.len() < self.n_batches() + 1 {
            return Err(Error::msg(format!(
                "Not enough values ({}) to bound {} filter sets.",
                input.len(),
                self.n_batches()
            )));
        }
        Ok(())
    }

    /// Get the number of batches.
    fn n_batches(&self) -> usize {
        self.filters.len()
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
    ///     This is only need it the reference file needed is not the standard
    ///     muon nexus v2 file. The ref_file is generated from tools/make_default.py.
    pub fn save_nexus(&self, filename: String, ref_file: String) -> Result<()> {
        // 1. Read p_info from input file
        let dataset = self.dataset.as_ref().unwrap();
        let (periods, dwell) = get_period_info(&dataset.filename)?;

        // 2. Setup shapes map
        let mut shapes = std::collections::HashMap::new();
        let n = dataset.n_spec;
        shapes.insert("N".to_string(), n);
        shapes.insert("P".to_string(), periods);
        shapes.insert("NP".to_string(), n * periods);
        shapes.insert("PD".to_string(), periods + dwell);
        shapes.insert("NPD".to_string(), n * (periods + dwell));

        // 3. Run save_default to merge/copy from ref_file
        save_default(&filename, &ref_file, &shapes)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::MockData;
    use ndarray::Array1;

    /// Build a BatchData with `n` filter sets using MockData as the
    /// underlying dataset (no real .nxs file needed).
    fn make_batch(n_filter_sets: usize) -> BatchData {
        let mock = MockData::new().unwrap();
        let dataset = mock.create(64, 1048576).unwrap();
        BatchData {
            dataset: Some(dataset),
            results: (0..n_filter_sets)
                .map(|_| Histogram::new(0, 32768, 2048))
                .collect(),
            filters: (0..n_filter_sets).map(|_| Filters::new()).collect(),
            labels: vec![None; n_filter_sets],
            data_changed: vec![true; n_filter_sets],
        }
    }

    /// resolve_indices(All) should return every index in range.
    #[test]
    fn test_resolve_indices_all() {
        let batch = make_batch(3);
        let indices = batch.resolve_indices(&FilterIndex::All).unwrap();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    /// resolve_indices(Index) should return a single valid index.
    #[test]
    fn test_resolve_indices_single_valid() {
        let batch = make_batch(3);
        let indices = batch.resolve_indices(&FilterIndex::Index(1)).unwrap();
        assert_eq!(indices, vec![1]);
    }

    /// resolve_indices(Index) should error for an out-of-range index.
    #[test]
    fn test_resolve_indices_out_of_range() {
        let batch = make_batch(3);
        let result = batch.resolve_indices(&FilterIndex::Index(3));
        assert!(result.is_err());
    }

    /// resolve_indices(Label) should return the index of the labelled set.
    #[test]
    fn test_resolve_indices_by_label() {
        let mut batch = make_batch(3);
        batch
            .set_label(1, Some("my_favourite".to_string()))
            .unwrap();

        let indices = batch
            .resolve_indices(&FilterIndex::Label("my_favourite".to_string()))
            .unwrap();
        assert_eq!(indices, vec![1]);
    }

    /// A label already used by another filter set should be rejected.
    #[test]
    fn test_duplicate_label_rejected() {
        let mut batch = make_batch(3);
        batch
            .set_label(0, Some("my_favourite".to_string()))
            .unwrap();

        assert!(batch
            .set_label(1, Some("my_favourite".to_string()))
            .is_err());
        // the failed call should not have changed anything
        assert_eq!(
            batch.labels(),
            vec![Some("my_favourite".to_string()), None, None]
        );
    }

    /// 'all' is reserved for selecting every filter set, in any case.
    #[test]
    fn test_all_rejected_as_label() {
        let mut batch = make_batch(2);
        assert!(batch.set_label(0, Some("all".to_string())).is_err());
        assert!(batch.set_label(0, Some("ALL".to_string())).is_err());
    }

    /// An empty label cannot be used, as it names nothing.
    #[test]
    fn test_empty_label_rejected() {
        let mut batch = make_batch(2);
        assert!(batch.set_label(0, Some(String::new())).is_err());
    }

    /// Labelling an out-of-range filter set should error.
    #[test]
    fn test_set_label_out_of_range() {
        let mut batch = make_batch(2);
        assert!(batch
            .set_label(5, Some("my_favourite".to_string()))
            .is_err());
    }

    /// Relabelling a filter set should drop its previous label, rather than
    /// leaving the set addressable by both.
    #[test]
    fn test_relabelling_drops_old_label() {
        let mut batch = make_batch(2);
        batch.set_label(0, Some("old".to_string())).unwrap();
        batch.set_label(0, Some("new".to_string())).unwrap();

        assert_eq!(batch.labels(), vec![Some("new".to_string()), None]);
        assert!(batch
            .resolve_indices(&FilterIndex::Label("old".to_string()))
            .is_err());
    }

    /// Filter-mutating methods should accept a label in place of an index.
    #[test]
    fn test_add_time_filter_by_label() {
        let mut batch = make_batch(3);
        batch
            .set_label(1, Some("my_favourite".to_string()))
            .unwrap();
        batch
            .add_time_filter(
                FilterIndex::Label("my_favourite".to_string()),
                "f1".to_string(),
                1.0,
                2.0,
            )
            .unwrap();

        let (starts0, _) = batch.filters[0].get_time_filter_times();
        let (starts1, ends1) = batch.filters[1].get_time_filter_times();
        let (starts2, _) = batch.filters[2].get_time_filter_times();

        assert!(starts0.is_empty());
        assert_eq!(starts1, vec![1e9 as usize]);
        assert_eq!(ends1, vec![2e9 as usize]);
        assert!(starts2.is_empty());
    }

    /// Results should be addressable by label too, via the same mechanism.
    #[test]
    fn test_get_n_events_by_label() {
        let mut batch = make_batch(3);
        batch
            .set_label(2, Some("my_favourite".to_string()))
            .unwrap();
        // set distinct counts so the lookup can't pass by picking the wrong set
        batch.results[2].n = 42;

        let by_label = batch
            .get_n_events(FilterIndex::Label("my_favourite".to_string()))
            .unwrap();
        assert_eq!(by_label, vec![42]);
        assert_eq!(by_label, batch.get_n_events(FilterIndex::Index(2)).unwrap());
    }

    /// Concatenation preserves labels, shifting them with their filter sets.
    #[test]
    fn test_concatenate_preserves_and_shifts_labels() {
        let mut left = make_batch(2);
        left.set_label(1, Some("left".to_string())).unwrap();
        let mut right = make_batch(2);
        right.set_label(0, Some("right".to_string())).unwrap();

        let combined = left.concatenate(right).unwrap();

        assert_eq!(
            combined.labels(),
            vec![
                None,
                Some("left".to_string()),
                Some("right".to_string()),
                None
            ]
        );
        // 'right' was index 0 of its own object, and is index 2 of the result
        assert_eq!(
            combined
                .resolve_indices(&FilterIndex::Label("right".to_string()))
                .unwrap(),
            vec![2]
        );
    }

    /// Concatenating two objects that use the same label should error, as
    /// labels must stay unique.
    #[test]
    fn test_concatenate_duplicate_label_errors() {
        let mut left = make_batch(2);
        left.set_label(0, Some("sweep".to_string())).unwrap();
        let mut right = make_batch(2);
        right.set_label(1, Some("sweep".to_string())).unwrap();

        assert!(left.concatenate(right).is_err());
    }

    /// add() merges two filter sets into one, so the result corresponds to
    /// neither input and labels are dropped.
    #[test]
    fn test_add_drops_labels() {
        let mut left = make_batch(2);
        left.set_label(0, Some("left".to_string())).unwrap();
        let mut right = make_batch(2);
        right.set_label(1, Some("right".to_string())).unwrap();

        let combined = left.add(right).unwrap();
        assert_eq!(combined.labels(), vec![None, None]);
    }

    /// combinations() builds each filter set from a pair, so the result
    /// corresponds to neither input and labels are dropped.
    #[test]
    fn test_combinations_drops_labels() {
        let mut left = make_batch(2);
        left.set_label(0, Some("left".to_string())).unwrap();
        let mut right = make_batch(3);
        right.set_label(1, Some("right".to_string())).unwrap();

        let combined = left.combinations(right).unwrap();
        assert_eq!(combined.labels(), vec![None; 6]);
    }

    /// __repr__ should show a label when there is one, and just the index
    /// when there is not.
    #[test]
    fn test_repr_shows_labels() {
        let mut batch = make_batch(2);
        batch
            .set_label(1, Some("my_favourite".to_string()))
            .unwrap();

        let repr = batch.__repr__();
        assert!(repr.contains("Filter set 0:"));
        assert!(repr.contains("Filter set 1 (my_favourite):"));
    }

    /// BatchData::new should error when n_filter_sets is 0.
    #[test]
    fn test_new_zero_filter_sets_errors() {
        let result = BatchData::new("dummy.nxs".to_string(), 64, 0, 1048576);
        assert!(result.is_err());
    }

    /// Adding a time filter at a single index should only affect that
    /// filter set, leaving the others untouched.
    #[test]
    fn test_add_time_filter_single_index() {
        let mut batch = make_batch(3);
        batch
            .add_time_filter(FilterIndex::Index(1), "f1".to_string(), 1.0, 2.0)
            .unwrap();

        let (starts0, ends0) = batch.filters[0].get_time_filter_times();
        let (starts1, ends1) = batch.filters[1].get_time_filter_times();
        let (starts2, ends2) = batch.filters[2].get_time_filter_times();

        assert!(starts0.is_empty() && ends0.is_empty());
        assert_eq!(starts1, vec![1e9 as usize]);
        assert_eq!(ends1, vec![2e9 as usize]);
        assert!(starts2.is_empty() && ends2.is_empty());
    }

    /// Adding a time filter with index "All" should apply it to every
    /// filter set.
    #[test]
    fn test_add_time_filter_all_indices() {
        let mut batch = make_batch(3);
        batch
            .add_time_filter(FilterIndex::All, "f1".to_string(), 1.0, 2.0)
            .unwrap();

        for filters in &batch.filters {
            let (starts, ends) = filters.get_time_filter_times();
            assert_eq!(starts, vec![1e9 as usize]);
            assert_eq!(ends, vec![2e9 as usize]);
        }
    }

    /// Adding a time filter at an out-of-range index should error and not
    /// modify any filter set.
    #[test]
    fn test_add_time_filter_out_of_range() {
        let mut batch = make_batch(3);
        batch
            .add_time_filter(FilterIndex::Index(1), "good filter".to_string(), 0.5, 1.5)
            .unwrap();
        let result = batch.add_time_filter(FilterIndex::Index(5), "f1".to_string(), 1.0, 2.0);
        assert!(result.is_err());

        for (index, filters) in batch.filters.into_iter().enumerate() {
            let (starts, ends) = filters.get_time_filter_times();
            if index == 1 {
                // converted to ns
                assert_eq!(starts, vec![0.5e9 as usize]);
                assert_eq!(ends, vec![1.5e9 as usize]);
            } else {
                assert!(starts.is_empty());
            }
        }
    }

    /// Removing a time filter at a single index should only affect that
    /// filter set.
    #[test]
    fn test_remove_time_filter_single_index() {
        let mut batch = make_batch(2);
        batch
            .add_time_filter(FilterIndex::All, "f1".to_string(), 1.0, 2.0)
            .unwrap();
        batch
            .add_time_filter(FilterIndex::Index(0), "f2".to_string(), 2.5, 3.5)
            .unwrap();

        batch
            .remove_time_filter(FilterIndex::Index(0), "f1".to_string())
            .unwrap();

        let (starts0, _) = batch.filters[0].get_time_filter_times();
        let (starts1, _) = batch.filters[1].get_time_filter_times();

        // converted to ns
        assert_eq!(starts0, vec![2.5e9 as usize]);
        assert_eq!(starts1, vec![1e9 as usize]);
    }

    /// Removing a time filter with index "All" should remove it from
    /// every filter set.
    #[test]
    fn test_remove_time_filter_all_indices() {
        let mut batch = make_batch(2);
        batch
            .add_time_filter(FilterIndex::All, "f1".to_string(), 1.0, 2.0)
            .unwrap();
        batch
            .add_time_filter(FilterIndex::Index(0), "f2".to_string(), 2.5, 3.5)
            .unwrap();

        batch
            .remove_time_filter(FilterIndex::All, "f1".to_string())
            .unwrap();

        let (starts0, _) = batch.filters[0].get_time_filter_times();
        let (starts1, _) = batch.filters[1].get_time_filter_times();

        assert_eq!(starts0, vec![2.5e9 as usize]);
        assert!(starts1.is_empty())
    }

    /// array time filters should give each filter set one time filter.
    #[test]
    fn test_array_to_time_filters() {
        let mut batch = make_batch(4);
        let array = Array1::from_vec(vec![1., 3., 4., 5., 10.]);
        batch
            .array_to_time_filters("f1".to_string(), array.clone())
            .unwrap();

        for (i, filters) in batch.filters.iter().enumerate() {
            let (starts, ends) = filters.get_time_filter_times();
            assert_eq!(starts, vec![(array[i] as f64 * 1e9) as usize]);
            assert_eq!(ends, vec![(array[i + 1] as f64 * 1e9) as usize]);
        }
    }

    /// array log filters should give each filter set one sample log filter.
    #[test]
    fn test_array_log_filters() {
        let mut batch = make_batch(4);
        let array = Array1::from_vec(vec![1., 3., 4., 5., 10.]);
        batch
            .array_to_log_filters("lf1".to_string(), "temp".to_string(), array.clone())
            .unwrap();

        for (k, filters) in batch.filters.into_iter().enumerate() {
            assert_eq!(filters.get_required_log_names(), vec!["temp".to_string()]);
            assert_eq!(filters.sample_log_filters[0].lower, Some(array[k]));
            assert_eq!(filters.sample_log_filters[0].upper, Some(array[k + 1]));
        }
    }

    /// Adding a log filter at a single index should only affect that
    /// filter set.
    #[test]
    fn test_add_log_filter_single_index() {
        let mut batch = make_batch(3);
        batch
            .add_log_filter(
                FilterIndex::Index(2),
                "lf1".to_string(),
                "temp".to_string(),
                1.0,
                2.0,
            )
            .unwrap();

        assert!(batch.filters[0].get_required_log_names().is_empty());
        assert!(batch.filters[1].get_required_log_names().is_empty());
        assert_eq!(
            batch.filters[2].get_required_log_names(),
            vec!["temp".to_string()]
        );
    }

    /// Adding a log filter with index "All" should apply it to every
    /// filter set.
    #[test]
    fn test_add_log_filter_all_indices() {
        let mut batch = make_batch(3);
        batch
            .add_log_filter(
                FilterIndex::All,
                "lf1".to_string(),
                "temp".to_string(),
                1.0,
                2.0,
            )
            .unwrap();

        for filters in &batch.filters {
            assert_eq!(filters.get_required_log_names(), vec!["temp".to_string()]);
        }
    }

    /// Removing a log filter at a single index should only affect that
    /// filter set.
    #[test]
    fn test_remove_log_filter_single_index() {
        let mut batch = make_batch(2);
        batch
            .add_log_filter(
                FilterIndex::All,
                "lf1".to_string(),
                "temp".to_string(),
                1.0,
                2.0,
            )
            .unwrap();

        batch
            .remove_log_filter(FilterIndex::Index(1), "lf1".to_string())
            .unwrap();

        assert_eq!(
            batch.filters[0].get_required_log_names(),
            vec!["temp".to_string()]
        );
        assert!(batch.filters[1].get_required_log_names().is_empty());
    }

    /// Removing a log filter with index "All" should remove it from every
    /// filter set.
    #[test]
    fn test_remove_log_filter_all_indices() {
        let mut batch = make_batch(3);
        batch
            .add_log_filter(
                FilterIndex::All,
                "lf1".to_string(),
                "temp".to_string(),
                1.0,
                2.0,
            )
            .unwrap();

        batch
            .remove_log_filter(FilterIndex::All, "lf1".to_string())
            .unwrap();

        for filters in &batch.filters {
            assert!(filters.get_required_log_names().is_empty());
        }
    }

    /// Setting an amplitude filter at a single index should only affect
    /// that filter set's amplitude array.
    #[test]
    fn test_set_amp_single_index() {
        let mut batch = make_batch(2);
        batch.set_amp(FilterIndex::Index(1), 3, 5.0).unwrap();

        let amps0 = batch.filters[0].get_amps(6).unwrap();
        let amps1 = batch.filters[1].get_amps(6).unwrap();

        assert_eq!(amps0, Array1::<f64>::zeros(6));
        assert_eq!(
            amps1,
            ndarray::Array1::from_vec(vec![0., 0., 0., 5., 0., 0.])
        );
    }

    /// Setting an amplitude filter with index "All" should apply it to
    /// every filter set.
    #[test]
    fn test_set_amp_all_indices() {
        let mut batch = make_batch(3);
        batch.set_amp(FilterIndex::All, 2, 4.4).unwrap();

        for filters in &batch.filters {
            let amps = filters.get_amps(6).unwrap();
            assert_eq!(
                amps,
                ndarray::Array1::from_vec(vec![0., 0., 4.4, 0., 0., 0.])
            );
        }
    }

    /// Concatenating should append the other object's filter sets and
    /// results, keeping the order of both.
    #[test]
    fn test_concatenate() {
        let mut batch = make_batch(2);
        batch
            .add_time_filter(FilterIndex::All, "a".to_string(), 1.0, 2.0)
            .unwrap();
        let mut other = make_batch(3);
        other
            .add_time_filter(FilterIndex::All, "b".to_string(), 3.0, 4.0)
            .unwrap();

        let combined = batch.concatenate(other).unwrap();

        assert_eq!(combined.__len__(), 5);
        assert_eq!(combined.results.len(), 5);
        for (i, filters) in combined.filters.iter().enumerate() {
            let (starts, ends) = filters.get_time_filter_times();
            if i < 2 {
                assert_eq!(starts, vec![1e9 as usize]);
                assert_eq!(ends, vec![2e9 as usize]);
            } else {
                assert_eq!(starts, vec![3e9 as usize]);
                assert_eq!(ends, vec![4e9 as usize]);
            }
        }
    }

    /// Adding should merge the filter sets of both objects pairwise,
    /// leaving the number of filter sets unchanged.
    #[test]
    fn test_add() {
        let mut batch = make_batch(3);
        for i in 0..3 {
            batch
                .add_time_filter(
                    FilterIndex::Index(i),
                    format!("a{i}"),
                    i as f64,
                    i as f64 + 1.,
                )
                .unwrap();
        }

        let mut other = make_batch(3);
        for j in 0..3 {
            other
                .add_log_filter(
                    FilterIndex::Index(j),
                    format!("b{j}"),
                    format!("log{j}"),
                    0.,
                    1.,
                )
                .unwrap();
        }

        let combined = batch.add(other).unwrap();

        assert_eq!(combined.__len__(), 3);
        assert_eq!(combined.results.len(), 3);
        assert_eq!(combined.data_changed, vec![true; 3]);

        for (i, filters) in combined.filters.iter().enumerate() {
            // the time filter comes from this object's filter set i
            let (starts, ends) = filters.get_time_filter_times();
            assert_eq!(starts, vec![(i as f64 * 1e9) as usize]);
            assert_eq!(ends, vec![((i + 1) as f64 * 1e9) as usize]);
            // the log filter comes from the other object's filter set i
            assert_eq!(filters.get_required_log_names(), vec![format!("log{i}")]);
        }
    }

    /// Adding objects with different numbers of filter sets should error.
    #[test]
    fn test_add_mismatched_lengths() {
        let batch = make_batch(2);
        let other = make_batch(3);

        let error = batch.add(other).err().unwrap();

        assert_eq!(
            error.to_string(),
            "Can only add data objects with the same number of batches.".to_string()
        );
    }

    /// Concatenating should keep each object's own histogram settings.
    #[test]
    fn test_concatenate_keeps_histogram_settings() {
        let mut batch = make_batch(1);
        batch
            .set_histogram_settings(FilterIndex::All, 0., 1., 10)
            .unwrap();
        let mut other = make_batch(1);
        other
            .set_histogram_settings(FilterIndex::All, 0., 2., 20)
            .unwrap();

        let combined = batch.concatenate(other).unwrap();

        assert_eq!(
            (
                combined.results[0].min_time,
                combined.results[0].max_time,
                combined.results[0].n_bins
            ),
            (0, 1000, 10)
        );
        assert_eq!(
            (
                combined.results[1].min_time,
                combined.results[1].max_time,
                combined.results[1].n_bins
            ),
            (0, 2000, 20)
        );
    }

    /// Combining should produce the cartesian product of both filter sets,
    /// ordered with the other object's filter sets varying fastest.
    #[test]
    fn test_combinations() {
        let mut batch = make_batch(2);
        batch
            .add_time_filter(FilterIndex::Index(0), "a0".to_string(), 1.0, 2.0)
            .unwrap();
        batch
            .add_time_filter(FilterIndex::Index(1), "a1".to_string(), 3.0, 4.0)
            .unwrap();

        let mut other = make_batch(3);
        for j in 0..3 {
            other
                .add_log_filter(
                    FilterIndex::Index(j),
                    format!("b{j}"),
                    format!("log{j}"),
                    0.,
                    1.,
                )
                .unwrap();
        }

        let combined = batch.combinations(other).unwrap();

        assert_eq!(combined.__len__(), 6);
        assert_eq!(combined.results.len(), 6);
        assert_eq!(combined.data_changed, vec![true; 6]);

        for i in 0..2 {
            for j in 0..3 {
                let k = i * 3 + j;
                let filters = &combined.filters[k];
                // the time filter comes from this object's filter set i
                let (starts, ends) = filters.get_time_filter_times();
                assert_eq!(starts, vec![((2 * i + 1) as f64 * 1e9) as usize]);
                assert_eq!(ends, vec![((2 * i + 2) as f64 * 1e9) as usize]);
                // the log filter comes from the other object's filter set j
                assert_eq!(filters.get_required_log_names(), vec![format!("log{j}")]);
            }
        }
    }

    /// Combining with a single-filter-set object should leave the number of
    /// filter sets unchanged.
    #[test]
    fn test_combinations_with_single_filter_set() {
        let mut batch = make_batch(3);
        batch
            .add_time_filter(FilterIndex::All, "a".to_string(), 1.0, 2.0)
            .unwrap();
        let mut other = make_batch(1);
        other
            .add_log_filter(
                FilterIndex::All,
                "b".to_string(),
                "temp".to_string(),
                0.,
                1.,
            )
            .unwrap();

        let combined = batch.combinations(other).unwrap();

        assert_eq!(combined.__len__(), 3);
        for filters in &combined.filters {
            let (starts, _) = filters.get_time_filter_times();
            assert_eq!(starts, vec![1e9 as usize]);
            assert_eq!(filters.get_required_log_names(), vec!["temp".to_string()]);
        }
    }

    /// Combining objects with the same dataset should keep that dataset.
    #[test]
    fn test_combine_data_same_dataset() {
        let batch = make_batch(1);
        let other = make_batch(1);

        let dataset = batch.combine_data(&other.dataset).unwrap();

        assert_eq!(
            dataset.unwrap().filename,
            batch.dataset.as_ref().unwrap().filename
        );
    }

    /// Combining an object holding data with one holding none should keep
    /// the data, whichever side it is on.
    #[test]
    fn test_combine_data_one_none() {
        let batch = make_batch(1);
        let empty = BatchData::empty(1);

        let dataset = batch.combine_data(&empty.dataset).unwrap();
        assert!(dataset.is_some());

        let dataset = empty.combine_data(&batch.dataset).unwrap();
        assert_eq!(
            dataset.unwrap().filename,
            batch.dataset.as_ref().unwrap().filename
        );
    }

    /// Combining two objects with no data should give no data.
    #[test]
    fn test_combine_data_both_none() {
        let batch = BatchData::empty(1);
        let other = BatchData::empty(1);

        let dataset = batch.combine_data(&other.dataset).unwrap();

        assert!(dataset.is_none());
    }

    /// Combining objects with different datasets should error.
    #[test]
    fn test_combine_data_different_datasets() {
        let batch = make_batch(1);
        let mut other = make_batch(1);
        other.dataset.as_mut().unwrap().filename = "somewhere_else.nxs".to_string();

        let error = batch.combine_data(&other.dataset).err().unwrap();

        assert_eq!(
            error.to_string(),
            "Tried to combine BatchData objects with different data!".to_string()
        );
    }

    /// Concatenating an object with data onto one without should carry the
    /// data over to the result.
    #[test]
    fn test_concatenate_takes_other_dataset() {
        let batch = BatchData::empty(2);
        let other = make_batch(1);

        let combined = batch.concatenate(other).unwrap();

        assert!(combined.dataset.is_some());
        assert_eq!(combined.__len__(), 3);
    }

    /// Combining an object with data onto one without should carry the data
    /// over to the result.
    #[test]
    fn test_combinations_takes_other_dataset() {
        let batch = BatchData::empty(2);
        let other = make_batch(3);

        let combined = batch.combinations(other).unwrap();

        assert!(combined.dataset.is_some());
        assert_eq!(combined.__len__(), 6);
    }

    /// Setting the time filter type at a single index should only affect
    /// that filter set.
    #[test]
    fn test_set_time_type_single_index() {
        let mut batch = make_batch(2);
        batch
            .set_time_type(FilterIndex::Index(1), "exclude".to_string())
            .unwrap();

        assert!(batch.filters[0].is_include());
        assert!(!batch.filters[1].is_include());
    }

    /// Setting the time filter type with index "All" should apply it to
    /// every filter set.
    #[test]
    fn test_set_time_type_all_indices() {
        let mut batch = make_batch(3);
        batch
            .set_time_type(FilterIndex::All, "exclude".to_string())
            .unwrap();

        for filters in &batch.filters {
            assert!(!filters.is_include());
        }
    }
}
