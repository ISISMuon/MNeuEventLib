use anyhow::{Error, Result};
use pyo3::prelude::{Borrowed, pyclass, pymethods, FromPyObject, PyAny};
use pyo3::types::{PyInt, PyString};

use crate::data::{NexusData};
use crate::filters::Filters;
use crate::stats::Histogram;

/// The index of a filter set (and its corresponding result) within a
/// [`BatchData`], or a request to apply an operation to every filter set.
///
/// From Python this can be constructed from either an integer (e.g. `0`,
/// `1`, ...) or the string `"all"` (case-insensitive).
pub enum FilterIndex {
    /// Apply the operation to every filter set.
    All,
    /// Apply the operation to a single filter set at this index.
    Index(usize),
}

impl<'a, 'py> FromPyObject<'a, 'py> for FilterIndex {
    type Error = Error;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self> {
        // If index is given as an integer, turn into Index integer
        if let Ok(index) = obj.cast::<PyInt>() {
            return Ok(FilterIndex::Index(index.extract()?))
        // If index is given as a string, check it is 'all' or fail
        } else if let Ok(string) = obj.cast::<PyString>() {
            if string.extract::<String>()?.to_lowercase() == "all" {
                return Ok(FilterIndex::All);
            }
        }
        // If index is anything else, fail
        Err(Error::msg("Filter index must be a number or 'all'"))
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
/// set) or `i` (apply the change to filter set `i`
/// only).
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct BatchData {
    #[pyo3(get)]
    pub dataset: NexusData,
    pub results: Vec<Histogram>,
    pub filters: Vec<Filters>,
    data_changed: Vec<bool>, // whether data has changed since last calculation, per filter set
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
            dataset,
            results: (0..n_filter_sets)
                .map(|_| Histogram::new(0., 32.768, 2048))
                .collect(),
            filters: (0..n_filter_sets).map(|_| Filters::new()).collect(),
            data_changed: vec![true; n_filter_sets],
        })
    }

    /// The number of filter sets (and results) held by this object.
    pub fn n_filter_sets(&self) -> usize {
        self.filters.len()
    }

    /// Calculate the histograms for the current data and each filter set.
    ///
    /// Returns
    /// -------
    /// BatchData
    ///     This object, with `results[i]` holding the histogram calculated
    ///     from `dataset` and `filters[i]`, for each `i`.
    pub fn calculate(&mut self) -> Result<BatchData> {
        for i in 0..self.filters.len() {
            if self.data_changed[i] {
                self.data_changed[i] = false;
                let result = self.results[i].calculate(&self.dataset, &self.filters[i])?;
                self.results[i] = result;
            }
        }
        Ok(self.clone())
    }

    /// Set histogram settings for one or all filter sets.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// min_time: float
    ///     The minimum time bound for the histogram.
    /// max_time: float
    ///     The maximum time bound for the histogram.
    /// n_bins: int
    ///     The number of bins to divide the time range into.
    fn set_histogram_settings(
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
            self.results[i] = Histogram::new(min_time, max_time, n_bins);
        }
        Ok(())
    }

    /// Set the type for the time filters.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// filter_type: str
    ///     The type for the time filters. Must be 'exclude' or 'include'.
    fn set_time_type(&mut self, index: FilterIndex, filter_type: String) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.filters[i].set_time_type(filter_type.clone())?;
        }
        Ok(())
    }

    /// Add a time filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
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
            self.data_changed[i] = true;
            self.filters[i].add_time_filter(name.clone(), start, end)?;
        }
        Ok(())
    }

    /// Remove a time filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// name: str
    ///     The name of the time filter to remove.
    fn remove_time_filter(&mut self, index: FilterIndex, name: String) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.filters[i].remove_time_filter(name.clone())?;
        }
        Ok(())
    }

    /// Add a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
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
            self.data_changed[i] = true;
            self.filters[i].add_log_filter(name.clone(), log.clone(), Some(lower), Some(upper))?;
        }
        Ok(())
    }

    /// Remove a sample log filter.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// name: str
    ///     The name of the log filter to remove.
    fn remove_log_filter(&mut self, index: FilterIndex, name: String) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.filters[i].remove_log_filter(name.clone())?;
        }
        Ok(())
    }

    /// Add a sample log filter for all data above a certain value.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// name: str
    ///     The name of the log filter. Must be unique within each modified filter set.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// lower: float
    ///     The lower bound for the log filter.
    fn add_log_filter_above(
        &mut self,
        index: FilterIndex,
        name: String,
        log: String,
        lower: f64,
    ) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.filters[i].add_log_filter_above(name.clone(), log.clone(), lower)?;
        }
        Ok(())
    }

    /// Add a sample log filter for all data below a certain value.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// name: str
    ///     The name of the log filter. Must be unique within each modified filter set.
    /// log: str
    ///     The sample log in the data to which the filter applies.
    /// upper: float
    ///     The upper bound for the log filter.
    fn add_log_filter_below(
        &mut self,
        index: FilterIndex,
        name: String,
        log: String,
        upper: f64,
    ) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.filters[i].add_log_filter_below(name.clone(), log.clone(), upper)?;
        }
        Ok(())
    }

    /// Set the amplitude filter for a detector.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// detector: int
    ///     The detector to set a filter for.
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    fn set_amp(&mut self, index: FilterIndex, detector: usize, amp: f64) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.filters[i].set_amp(detector, amp);
        }
        Ok(())
    }

    /// Set an amplitude filter for all detectors that don't have one defined.
    ///
    /// Parameters
    /// ----------
    /// index: int | str
    ///     Either 'all', or the index of the filter set to modify.
    /// amp: float
    ///     The maximum amplitude that should be ignored.
    fn set_amps_baseline(&mut self, index: FilterIndex, amp: f64) -> Result<()> {
        for i in self.resolve_indices(&index)? {
            self.data_changed[i] = true;
            self.filters[i].set_amps_baseline(amp);
        }
        Ok(())
    }

    /// Save a filter set's result to a file.
    ///
    /// Parameters
    /// ----------
    /// index: int
    ///     The index of the filter set/result to save. Must be a single
    ///     filter set, not 'all'.
    /// filename: str
    ///     The filename for the saved file.
    fn save(&self, index: usize, filename: String) -> Result<()> {
        todo!();
        Ok(())
    }

    fn __repr__(&self) -> String {
        let mut string = self.dataset.__repr__();
        for (i, (filters, results)) in self.filters.iter().zip(self.results.iter()).enumerate() {
            string += &format!(
                "\n\nFilter set {i}:\n{}\n\n{}",
                filters.__repr__(),
                results.__repr__()
            );
        }
        string
    }
}

impl BatchData {
    /// Resolve a [`FilterIndex`] into a list of valid filter set indices,
    /// checking bounds along the way.
    fn resolve_indices(&self, index: &FilterIndex) -> Result<Vec<usize>> {
        match index {
            FilterIndex::All => Ok((0..self.filters.len()).collect()),
            FilterIndex::Index(i) => {
                self.check_index(*i)?;
                Ok(vec![*i])
            }
        }
    }

    /// Check that a given index is valid for this BatchData's filter sets.
    fn check_index(&self, index: usize) -> Result<()> {
        if index >= self.filters.len() {
            return Err(Error::msg(format!(
                "Index {index} out of range: only {} filter sets exist.",
                self.filters.len()
            )));
        }
        Ok(())
    }
}
