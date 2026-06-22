use std::fs::File;
/// The user-facing API for the filter objects.
use std::collections::HashMap;

use anyhow::{Error, Result};
use ndarray::Array1;
use pyo3::prelude::{pyclass, pymethods, Bound};
use pyo3::types::PyType;
use serde::{Deserialize, Serialize};

use crate::consts::S_TO_NS;
use crate::data::SampleLog;

use std::io::Read;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum FilterType {
    Include,
    Exclude,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Filter {
    name: String,
    start: f64,
    end: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogFilter {
    name: String,
    log: String,
    lower: f64,
    upper: f64,
}

#[pyclass(from_py_object)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Filters {
    time_filter_type: FilterType,
    time_filters: Vec<Filter>,
    sample_log_filters: Vec<LogFilter>,
    amplitudes: HashMap<usize, f64>,
}

impl Filters {
    /// Get the start and end points of each time filter.
    pub fn get_time_filter_times(&self) -> (Vec<usize>, Vec<usize>) {
        // note this just gets the intervals for each filter; whether
        // they're include or exclude filters is handled by get_weights
        if self.time_filters.is_empty() {
            return (Vec::new(), Vec::new());
        }
        (
            self.time_filters
                .iter()
                .map(|f| (f.start * S_TO_NS) as usize)
                .collect(),
            self.time_filters
                .iter()
                .map(|f| (f.end * S_TO_NS) as usize)
                .collect(),
        )
    }

    /// Get the start and end times for each log filter.
    pub fn get_log_filter_times(
        &self,
        logs: HashMap<String, SampleLog>,
    ) -> (Vec<usize>, Vec<usize>) {
        // get the value log for each required sample log
        // the zip/unzip is to convert it from
        // Vec<(usize, usize)> to (Vec<usize>, Vec<usize>)
        self.sample_log_filters
            .iter()
            .flat_map(|f| {
                let (s, e) = logs[&f.log].to_time_ranges(f.lower, f.upper);
                s.into_iter().zip(e)
            })
            .unzip()
    }

    // Get the relevant log for each log filter.
    pub fn get_required_log_names(&self) -> Vec<String> {
        self.sample_log_filters
            .iter()
            .map(|f| f.log.clone())
            .collect()
    }

    /// Return whether the time filters are include or exclude.
    pub fn is_include(&self) -> bool {
        match self.time_filter_type {
            FilterType::Include => true,
            FilterType::Exclude => false,
        }
    }

    /// Turn the detectors map into an array for indexing.
    pub fn get_amps(&self, n_spec: usize) -> Result<Array1<f64>> {
        // usize::MAX is used as a placeholder key for 'all other detectors'
        let mut array: Array1<f64> = if let Some(val) = self.amplitudes.get(&usize::MAX) {
            Array1::from_elem(n_spec, *val)
        } else {
            Array1::zeros(n_spec)
        };
        for (key, value) in self.amplitudes.iter() {
            if key == &usize::MAX {
                continue;
            }
            if key >= &n_spec {
                return Err(Error::msg(format!("Attempted to set amplitude filter for detector {key}, but instrument only has {n_spec} detectors.")));
            }
            array[*key] = *value;
        }
        Ok(array)
    }
}

/// Internal of load function so we can call it from Rust.
fn _load(filename: String) -> Result<Filters> {
    let mut file = File::open(&filename)?;

    let mut contents = String::new();
    let _ = file.read_to_string(&mut contents);
    Ok(serde_json::from_str(contents.as_str())?)
}

#[pymethods]
impl Filters {
    #[new]
    pub fn new() -> Filters {
        Filters {
            time_filter_type: FilterType::Include,
            time_filters: Vec::<Filter>::new(),
            sample_log_filters: Vec::<LogFilter>::new(),
            amplitudes: HashMap::<usize, f64>::new(),
        }
    }

    /// Set the time filter type.
    pub fn set_time_type(&mut self, filter_type: String) -> Result<()> {
        match filter_type.to_lowercase().as_str() {
            "include" => {
                self.time_filter_type = FilterType::Include;
                Ok(())
            }
            "exclude" => {
                self.time_filter_type = FilterType::Exclude;
                Ok(())
            }
            _ => Err(Error::msg("Type must be 'include' or 'exclude'")),
        }
    }

    /// Add a time filter.
    pub fn add_time_filter(&mut self, name: String, start: f64, end: f64) -> Result<()> {
        // check name isn't already in use
        if self.time_filters.iter().any(|f| f.name == name) {
            return Err(Error::msg(
                "Name already exists! Use `Filters.report()` to see a list of all filters.",
            ));
        }
        if !start.is_finite() || !end.is_finite() {
            return Err(Error::msg("start and end must be finite."));
        }
        if start < 0.0 || end < 0.0 {
            return Err(Error::msg("start and end must be non-negative."));
        }
        if end <= start {
            return Err(Error::msg("end must be greater than start."));
        }
        self.time_filters.push(Filter { name, start, end });
        Ok(())
    }

    pub fn remove_time_filter(&mut self, name: String) -> Result<()> {
        match self.time_filters.iter().position(|f| f.name == name) {
            Some(i) => {
                self.time_filters.swap_remove(i);
                Ok(())
            }
            None => Err(Error::msg("No such name in time filters. Use `Filters.report()` to see a list of all filters.")),
        }
    }

    /// Add a log filter.
    pub fn add_log_filter(
        &mut self,
        name: String,
        log: String,
        lower: f64,
        upper: f64,
    ) -> Result<()> {
        if self.sample_log_filters.iter().any(|f| f.name == name) {
            return Err(Error::msg("Name already exists!"));
        }
        self.sample_log_filters.push(LogFilter {
            name,
            log,
            lower,
            upper,
        });
        Ok(())
    }

    pub fn remove_log_filter(&mut self, name: String) -> Result<()> {
        match self.sample_log_filters.iter().position(|f| f.name == name) {
            Some(i) => {
                self.sample_log_filters.swap_remove(i);
                Ok(())
            }
            None => Err(Error::msg("No such name in log filters.")),
        }
    }

    pub fn add_log_filter_above(&mut self, name: String, log: String, lower: f64) -> Result<()> {
        self.add_log_filter(name, log, lower, f64::INFINITY)
    }

    pub fn add_log_filter_below(&mut self, name: String, log: String, upper: f64) -> Result<()> {
        self.add_log_filter(name, log, -f64::INFINITY, upper)
    }

    /// Set the minimum amplitude for a given detector.
    pub fn set_amp(&mut self, detector: usize, amp: f64) {
        self.amplitudes.insert(detector, amp);
    }

    /// Set the minimum amplitude for all detectors not otherwise specified.
    pub fn set_amps_baseline(&mut self, amp: f64) {
        self.amplitudes.insert(usize::MAX, amp);
    }

    /// Save the filters to a JSON file.
    pub fn save(&self, filename: String) -> Result<()> {
        let file = File::create(&filename)?;
        Ok(serde_json::to_writer_pretty(file, &self)?)
    }

    /// Load filters from a JSON file.
    #[classmethod]
    pub fn load(_cls: &Bound<'_, PyType>, filename: String) -> Result<Filters> {
        _load(filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    use ndarray::Array1;

    use crate::data::ValueLog;

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
            sample_log_filters: Vec::<LogFilter>::new(),
            amplitudes: HashMap::new(),
        };

        let (starts, ends) = filters.get_time_filter_times();

        assert_eq!(starts, vec![1e9 as usize, 3e9 as usize, 5e9 as usize]);
        assert_eq!(ends, vec![2e9 as usize, 4e9 as usize, 6e9 as usize]);
    }

    /// Test filters objects are initialised correctly.
    #[test]
    fn test_new_filters_creates_empty_filters() {
        let filters = Filters::new();
        let (starts, ends) = filters.get_time_filter_times();
        assert_eq!(starts.len(), 0);
        assert_eq!(ends.len(), 0);
        assert!(filters.is_include()); // default is include
    }

    /// Test get_required_log_names returns the log name for each log filter.
    #[test]
    fn test_get_log_names() {
        let filters = Filters {
            time_filter_type: FilterType::Include,
            time_filters: Vec::<Filter>::new(),
            sample_log_filters: vec![
                LogFilter {
                    name: "a".to_string(),
                    log: "temp".to_string(),
                    lower: 1.,
                    upper: 2.,
                },
                LogFilter {
                    name: "b".to_string(),
                    log: "pulse_width".to_string(),
                    lower: 3.,
                    upper: 4.,
                },
                LogFilter {
                    name: "c".to_string(),
                    log: "pressure".to_string(),
                    lower: 5.,
                    upper: 6.,
                },
            ],
            amplitudes: HashMap::new(),
        };
        let logs = filters.get_required_log_names();
        assert_eq!(
            logs,
            vec![
                "temp".to_string(),
                "pulse_width".to_string(),
                "pressure".to_string()
            ]
        )
    }

    /// Test log_filter_times correctly gets the times from the log filters.
    #[test]
    fn test_log_filter_to_times() {
        // note we test individual time ranges in data::sample_logs, so we only need to test
        // that this routine correctly maps over log names (and multiple bands for one log)
        let filters = Filters {
            time_filter_type: FilterType::Include,
            time_filters: Vec::<Filter>::new(),
            sample_log_filters: vec![
                LogFilter {
                    name: "a".to_string(),
                    log: "simple".to_string(),
                    lower: 2.,
                    upper: 3.,
                },
                LogFilter {
                    name: "b".to_string(),
                    log: "simple".to_string(),
                    lower: 0.,
                    upper: 1.,
                },
                LogFilter {
                    name: "c".to_string(),
                    log: "complex".to_string(),
                    lower: 2.,
                    upper: 8.,
                },
            ],
            amplitudes: HashMap::new(),
        };

        // logs from data::sample_logs tests
        let simple_times = Array1::<f64>::linspace(0., 4., 4001);
        let simple_log = ValueLog::<f64> {
            time: simple_times.clone(),
            value: simple_times.clone(),
        };

        // add a sample log: f(t) = 2(t-3)^2 from 0 to 4
        let complex_times = Array1::<f64>::linspace(0., 6., 6001);
        let complex_log = ValueLog::<f64> {
            time: complex_times.clone(),
            value: Array1::<f64>::from_iter(complex_times.iter().map(|t| 2. * (t - 3.).powi(2))),
        };

        let mut logs = HashMap::<String, SampleLog>::new();
        logs.insert("simple".to_string(), SampleLog::F64(simple_log));
        logs.insert("complex".to_string(), SampleLog::F64(complex_log));

        let (starts, ends) = filters.get_log_filter_times(logs);
        let expected_starts = vec![2e9 as usize, 0, 1e9 as usize, 4e9 as usize];
        let expected_ends = vec![3e9 as usize, 1e9 as usize, 2e9 as usize, 5e9 as usize];
        assert_eq!(starts, expected_starts);
        assert_eq!(ends, expected_ends);
    }

    /// Test time filter type can correctly be set.
    #[test]
    fn test_set_time_type() {
        let mut filters = Filters::new();
        filters.set_time_type("include".to_string()).unwrap();
        assert!(filters.is_include());
        filters.set_time_type("exclude".to_string()).unwrap();
        assert!(!filters.is_include());
    }

    /// Test time filter type can correctly be set regardless of case.
    #[test]
    fn test_set_time_type_mixed_case() {
        let mut filters = Filters::new();
        filters.set_time_type("INCLudE".to_string()).unwrap();
        assert!(filters.is_include());
        filters.set_time_type("ExClUDe".to_string()).unwrap();
        assert!(!filters.is_include());
    }

    /// Test set_time_type gives an error when the type is not valid.
    #[test]
    fn test_set_time_type_invalid_type() {
        let mut filters = Filters::new();
        let result = filters.set_time_type("invalid".to_string());
        assert!(result.is_err());
        let result = filters.set_time_type("".to_string());
        assert!(result.is_err());
    }

    /// Test adding a time filter.
    #[test]
    fn test_add_single_time_filter() {
        let mut filters = Filters::new();
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
        let mut filters = Filters::new();
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
        let mut filters = Filters::new();
        filters
            .add_time_filter("filter1".to_string(), 1.0, 2.0)
            .unwrap();

        let result = filters.add_time_filter("filter1".to_string(), 3.0, 4.0);
        assert!(result.is_err());
    }

    /// Test time filters can be removed correctly.
    #[test]
    fn test_remove_time_filter() {
        let mut filters = Filters::new();
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
        let mut filters = Filters::new();
        let result = filters.remove_time_filter("nonexistent".to_string());
        assert!(result.is_err());
    }

    /// Test adding a single log filter.
    #[test]
    fn test_add_log_filter() {
        let mut filters = Filters::new();
        filters
            .add_log_filter("name".to_string(), "temp".to_string(), 0., 1.)
            .unwrap();
        assert_eq!(
            filters.sample_log_filters,
            vec![LogFilter {
                name: "name".to_string(),
                log: "temp".to_string(),
                lower: 0.,
                upper: 1.
            }]
        )
    }

    /// Test adding multiple log filters.
    #[test]
    fn test_add_multiple_log_filter() {
        let mut filters = Filters::new();
        filters
            .add_log_filter("name".to_string(), "temp".to_string(), 0., 1.)
            .unwrap();
        filters
            .add_log_filter_above("name2".to_string(), "p".to_string(), 5.)
            .unwrap();
        filters
            .add_log_filter_below("name3".to_string(), "temp".to_string(), 8.)
            .unwrap();
        assert_eq!(
            filters.sample_log_filters,
            vec![
                LogFilter {
                    name: "name".to_string(),
                    log: "temp".to_string(),
                    lower: 0.,
                    upper: 1.
                },
                LogFilter {
                    name: "name2".to_string(),
                    log: "p".to_string(),
                    lower: 5.,
                    upper: f64::INFINITY
                },
                LogFilter {
                    name: "name3".to_string(),
                    log: "temp".to_string(),
                    lower: -f64::INFINITY,
                    upper: 8.
                }
            ]
        )
    }

    /// Test an error is given when a log filter is given a duplicate name.
    #[test]
    fn test_add_log_filter_duplicate_name() {
        let mut filters = Filters::new();
        filters
            .add_log_filter("filter1".to_string(), "temp".to_string(), 1.0, 2.0)
            .unwrap();

        let result = filters.add_log_filter("filter1".to_string(), "temp".to_string(), 3.0, 4.0);
        assert!(result.is_err());
    }

    /// Test time filters can be removed correctly.
    #[test]
    fn test_remove_log_filter() {
        let mut filters = Filters::new();
        filters
            .add_log_filter("filter1".to_string(), "temp".to_string(), 1.0, 2.0)
            .unwrap();
        filters
            .add_log_filter("filter2".to_string(), "pw".to_string(), 3.0, 4.0)
            .unwrap();

        filters.remove_log_filter("filter1".to_string()).unwrap();

        assert_eq!(
            filters.sample_log_filters,
            vec![LogFilter {
                name: "filter2".to_string(),
                log: "pw".to_string(),
                lower: 3.,
                upper: 4.
            }]
        )
    }

    /// Test removing a time filter that doesn't exist throws an error.
    #[test]
    fn test_remove_log_filter_nonexistent() {
        let mut filters = Filters::new();
        let result = filters.remove_log_filter("nonexistent".to_string());
        assert!(result.is_err());
    }

    /// Test setting an amplitude for a given detector works.
    #[test]
    fn test_set_amp() {
        let mut filters = Filters::new();
        filters.set_amp(5, 3.);

        let mut expected = HashMap::<usize, f64>::new();
        expected.insert(5, 3.);
        assert_eq!(filters.amplitudes, expected)
    }

    /// Test setting an amplitude for all detectors works.
    #[test]
    fn test_set_amps_baseline() {
        let mut filters = Filters::new();
        filters.set_amps_baseline(3.);

        let mut expected = HashMap::<usize, f64>::new();
        expected.insert(usize::MAX, 3.);
        assert_eq!(filters.amplitudes, expected)
    }

    /// Test amplitudes are correctly converted to an array.
    #[test]
    fn test_get_amps() {
        let mut filters = Filters::new();
        filters.set_amp(2, 4.4);
        filters.set_amp(5, 9.);

        let expected = Array1::from_vec(vec![0., 0., 4.4, 0., 0., 9.]);
        let actual = filters.get_amps(6).unwrap();

        assert_eq!(actual, expected)
    }

    /// Test amplitudes are correctly converted to an array with baseline.
    #[test]
    fn test_get_amps_baseline() {
        let mut filters = Filters::new();
        filters.set_amp(2, 4.4);
        filters.set_amp(5, 9.);
        filters.set_amps_baseline(1.5);

        let expected = Array1::from_vec(vec![1.5, 1.5, 4.4, 1.5, 1.5, 9.]);
        let actual = filters.get_amps(6).unwrap();

        assert_eq!(actual, expected)
    }

    /// Test an error is produced if a bad amplitude is given.
    #[test]
    fn test_get_amps_bad_amp() {
        let mut filters = Filters::new();
        filters.set_amp(2, 4.4);
        filters.set_amp(20, 9.);

        let result = filters.get_amps(6);
        assert!(result.is_err());

        assert_eq!(result.unwrap_err().to_string(), "Attempted to set amplitude filter for detector 20, but instrument only has 6 detectors.".to_string())
    }

    /// Test that saving a filter and loading it back in works.
    #[test]
    fn test_save_load() {
        let mut filters = Filters {
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
            ],
            sample_log_filters: vec![LogFilter {
                name: "c".to_string(),
                log: "Temp".to_string(),
                lower: 5.,
                upper: 6.,
            }],
            amplitudes: HashMap::new(),
        };
        filters.amplitudes.insert(3, 5.);

        let mut temp_save_loc = temp_dir();
        temp_save_loc.push("filters.json");

        // these functions take strings but this is a PathBuf
        let tmp_path = temp_save_loc.to_string_lossy().to_string();

        let _ = filters.save(tmp_path.clone());
        let loaded_filters = _load(tmp_path).unwrap();

        assert_eq!(filters, loaded_filters)
    }
}
