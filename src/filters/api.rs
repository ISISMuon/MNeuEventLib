use pyo3::exceptions::PyValueError;
/// The user-facing API for the filter objects.
use pyo3::prelude::{pyclass, pymethods, PyResult};

#[derive(Clone)]
enum FilterType {
    Include,
    Exclude,
}

const S_TO_NS: f64 = 1e9;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Filters {
    time_filter_type: FilterType,
    time_filters: Vec<Filter>,
    sample_log_filters: Vec<Filter>,
    amplitudes: f64,
}

impl Filters {
    /// Get the start points of each filter.
    pub fn get_time_filter_starts(&self) -> Vec<usize> {
        self.time_filters
            .iter()
            .map(|f| (f.start * S_TO_NS) as usize)
            .collect()
    }

    /// Get the end points of each filter.
    pub fn get_time_filter_ends(&self) -> Vec<usize> {
        self.time_filters
            .iter()
            .map(|f| (f.end * S_TO_NS) as usize)
            .collect()
    }

    /// Return whether the time filters are include or exclude.
    pub fn is_include(&self) -> bool {
        match self.time_filter_type {
            FilterType::Include => true,
            FilterType::Exclude => false
        }
    }
}

#[pymethods]
impl Filters {
    #[new]
    fn new() -> PyResult<Filters> {
        Ok(Filters {
            time_filter_type: FilterType::Include,
            time_filters: Vec::<Filter>::new(),
            sample_log_filters: Vec::<Filter>::new(),
            amplitudes: 0.,
        })
    }

    /// Set the time filter type.
    fn set_time_type(&mut self, filter_type: String) -> PyResult<()> {
        match filter_type.to_lowercase().as_str() {
            "include" => {
                self.time_filter_type = FilterType::Include;
                Ok(())
            }
            "exclude" => {
                self.time_filter_type = FilterType::Exclude;
                Ok(())
            }
            _ => Err(PyValueError::new_err("Type must be 'include' or 'exclude'")),
        }
    }

    /// Add a time filter.
    fn add_time_filter(&mut self, name: String, start: f64, end: f64) -> PyResult<()> {
        // check name isn't already in use
        if self.time_filters.iter().any(|f| f.name == name) {
            return Err(PyValueError::new_err("Name already exists!"))
        }
        self.time_filters.push(Filter { name, start, end });
        Ok(())
    }

    fn remove_time_filter(&mut self, name: String) -> PyResult<()> {
        match self.time_filters.iter().position(|f| f.name == name) {
            Some(i) => {
                self.time_filters.swap_remove(i);
                Ok(())
            }
            None => Err(PyValueError::new_err("No such name in time filters.")),
        }
    }

    /// Add a log filter.
    fn add_log_filter(&mut self, name: String, start: f64, end: f64) -> PyResult<()> {
        self.sample_log_filters.push(Filter { name, start, end });
        Ok(())
    }

    fn remove_log_filter(&mut self, name: String) -> PyResult<()> {
        match self.time_filters.iter().position(|f| f.name == name) {
            Some(i) => {
                self.time_filters.swap_remove(i);
                Ok(())
            }
            None => Err(PyValueError::new_err("No such name in log filters.")),
        }
    }

    fn set_amp(&mut self, amp: f64) -> PyResult<()> {
        self.amplitudes = amp;
        Ok(())
    }
}

#[derive(Clone)]
struct Filter {
    name: String,
    start: f64,
    end: f64,
}
