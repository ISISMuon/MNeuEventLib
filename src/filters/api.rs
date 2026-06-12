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
            FilterType::Exclude => false,
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
            return Err(PyValueError::new_err("Name already exists!"));
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
        match self.sample_log_filters.iter().position(|f| f.name == name) {
            Some(i) => {
                self.sample_log_filters.swap_remove(i);
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

#[derive(Clone, Debug, PartialEq)]
struct Filter {
    name: String,
    start: f64,
    end: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test converting a Filters object to starts and ends.
    #[test]
    fn test_convert_filters() {
        let filters = Filters {
            time_filter_type: FilterType::Include,
            time_filters: vec![
                Filter {
                    name: "a".to_string(),
                    start: 1.,
                    end: 2.,
                },
                Filter {
                    name: "b".to_string(),
                    start: 3.,
                    end: 4.,
                },
                Filter {
                    name: "c".to_string(),
                    start: 5.,
                    end: 6.,
                },
            ],
            sample_log_filters: Vec::<Filter>::new(),
            amplitudes: 0.,
        };

        let starts = filters.get_time_filter_starts();
        let ends = filters.get_time_filter_ends();

        assert_eq!(starts, vec![1e9 as usize, 3e9 as usize, 5e9 as usize]);
        assert_eq!(ends, vec![2e9 as usize, 4e9 as usize, 6e9 as usize]);
    }

    /// Test filters objects are initialised correctly.
    #[test]
    fn test_new_filters_creates_empty_filters() {
        let filters = Filters::new().unwrap();
        assert_eq!(filters.get_time_filter_starts().len(), 0);
        assert_eq!(filters.get_time_filter_ends().len(), 0);
        assert!(filters.is_include()); // default is include
    }

    /// Test time filter type can correctly be set.
    #[test]
    fn test_set_time_type() {
        let mut filters = Filters::new().unwrap();
        filters.set_time_type("include".to_string()).unwrap();
        assert!(filters.is_include());
        filters.set_time_type("exclude".to_string()).unwrap();
        assert!(!filters.is_include());
    }

    /// Test set_time_type gives an error when the type is not valid.
    #[test]
    fn test_set_time_type_invalid_type() {
        let mut filters = Filters::new().unwrap();
        let result = filters.set_time_type("invalid".to_string());
        assert!(result.is_err());
        let result = filters.set_time_type("".to_string());
        assert!(result.is_err());
    }

    /// Test adding a time filter and converting it to filter starts and ends.
    #[test]
    fn test_add_single_time_filter() {
        let mut filters = Filters::new().unwrap();
        filters
            .add_time_filter("filter1".to_string(), 1.0, 2.0)
            .unwrap();

        assert_eq!(
            filters.time_filters,
            vec![Filter {
                name: "filter1".to_string(),
                start: 1.,
                end: 2.
            }]
        );
    }

    /// Test adding multiple time filters.
    #[test]
    fn test_add_multiple_time_filters() {
        let mut filters = Filters::new().unwrap();
        filters
            .add_time_filter("filter1".to_string(), 1.0, 2.0)
            .unwrap();
        filters
            .add_time_filter("filter2".to_string(), 3.0, 4.0)
            .unwrap();
        filters
            .add_time_filter("filter3".to_string(), 5.0, 6.0)
            .unwrap();

        assert_eq!(
            filters.time_filters,
            vec![
                Filter {
                    name: "filter1".to_string(),
                    start: 1.,
                    end: 2.
                },
                Filter {
                    name: "filter2".to_string(),
                    start: 3.,
                    end: 4.
                },
                Filter {
                    name: "filter3".to_string(),
                    start: 5.,
                    end: 6.
                },
            ]
        );
    }

    /// Test an error is given when a time filter is given a duplicate name.
    #[test]
    fn test_add_time_filter_duplicate_name() {
        let mut filters = Filters::new().unwrap();
        filters
            .add_time_filter("filter1".to_string(), 1.0, 2.0)
            .unwrap();

        let result = filters.add_time_filter("filter1".to_string(), 3.0, 4.0);
        assert!(result.is_err());
    }

    /// Test time filters can be removed correctly.
    #[test]
    fn test_remove_time_filter() {
        let mut filters = Filters::new().unwrap();
        filters
            .add_time_filter("filter1".to_string(), 1.0, 2.0)
            .unwrap();
        filters
            .add_time_filter("filter2".to_string(), 3.0, 4.0)
            .unwrap();

        filters.remove_time_filter("filter1".to_string()).unwrap();

        assert_eq!(
            filters.time_filters,
            vec![Filter {
                name: "filter2".to_string(),
                start: 3.,
                end: 4.
            }]
        )
    }

    /// Test removing a time filter that doesn't exist throws an error.
    #[test]
    fn test_remove_time_filter_nonexistent() {
        let mut filters = Filters::new().unwrap();
        let result = filters.remove_time_filter("nonexistent".to_string());
        assert!(result.is_err());
    }
}
