/// The user-facing API for the filter objects.
use std::collections::HashMap;
use std::fs::File;

use anyhow::{Error, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use tabled::{builder::Builder, Table, Tabled};

use crate::consts::S_TO_NS;
use crate::data::SampleLog;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum FilterType {
    Include,
    Exclude,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Tabled)]
pub struct Filter {
    name: String,
    start: f64,
    end: f64,
}

// LogFilter bounds are Options because serialisation doesn't support infinity or -infinity;
// we use None to represent those
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogFilter {
    name: String,
    log: String,
    lower: Option<f64>,
    upper: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Filters {
    time_filter_type: FilterType,
    time_filters: Vec<Filter>,
    sample_log_filters: Vec<LogFilter>,
    amplitudes: HashMap<usize, f64>,
}

impl Filters { 
    pub fn new() -> Filters {
        Filters {
            time_filter_type: FilterType::Include,
            time_filters: Vec::<Filter>::new(),
            sample_log_filters: Vec::<LogFilter>::new(),
            amplitudes: HashMap::<usize, f64>::new(),
        }
    }

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
                let (s, e) = logs[&f.log].to_time_ranges(
                    f.lower.unwrap_or(-f64::INFINITY),
                    f.upper.unwrap_or(f64::INFINITY),
                );
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
        lower: Option<f64>,
        upper: Option<f64>,
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
        self.add_log_filter(name, log, Some(lower), None)
    }

    pub fn add_log_filter_below(&mut self, name: String, log: String, upper: f64) -> Result<()> {
        self.add_log_filter(name, log, None, Some(upper))
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
        serde_json::to_writer_pretty(file, &self)?;
        Ok(())
    }

    /// Load filters from a JSON file.
    pub fn load(filename: String) -> Result<Filters> {
        let file = File::open(&filename)?;
        Ok(serde_json::from_reader(file)?)
    }

    /// Create a string describing the filter data.
    pub fn __repr__(&self) -> String {
        let times_table = Table::new(&self.time_filters);

        let mut log_builder = Builder::new();
        log_builder.push_record(["name", "log", "min", "max"]);
        for filter in &self.sample_log_filters {
            log_builder.push_record([
                &filter.name,
                &filter.log,
                &filter.lower.unwrap_or(-f64::INFINITY).to_string(),
                &filter.upper.unwrap_or(f64::INFINITY).to_string(),
            ]);
        }
        let log_table = log_builder.build();

        let mut amps_builder = Builder::new();
        amps_builder.push_record(["detector", "amplitude"]);
        for (detector, amp) in self.amplitudes.iter() {
            if detector == &usize::MAX {
                amps_builder.push_record(["baseline", &amp.to_string()]);
            } else {
                amps_builder.push_record([detector.to_string(), amp.to_string()]);
            }
        }
        let amps_table = amps_builder.build();
        let time_type = match self.time_filter_type {
            FilterType::Include => "include",
            FilterType::Exclude => "exclude",
        };
        format!("Time filter type: {time_type}\n\nTime filters:\n{times_table}\n\nLog filters:\n{log_table}\n\nAmplitude filters:\n{amps_table}")
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
                    lower: Some(1.),
                    upper: Some(2.),
                },
                LogFilter {
                    name: "b".to_string(),
                    log: "pulse_width".to_string(),
                    lower: Some(3.),
                    upper: Some(4.),
                },
                LogFilter {
                    name: "c".to_string(),
                    log: "pressure".to_string(),
                    lower: Some(5.),
                    upper: Some(6.),
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
                    lower: Some(2.),
                    upper: Some(3.),
                },
                LogFilter {
                    name: "b".to_string(),
                    log: "simple".to_string(),
                    lower: Some(0.),
                    upper: Some(1.),
                },
                LogFilter {
                    name: "c".to_string(),
                    log: "complex".to_string(),
                    lower: Some(2.),
                    upper: Some(8.),
                },
            ],
            amplitudes: HashMap::new(),
        };

        // logs from data::sample_logs tests
        let simple_times = Array1::<f64>::linspace(0., 4., 4001);
        let simple_log = ValueLog::<f64> {
            name: "simple".to_string(),
            time: simple_times.clone(),
            value: simple_times.clone(),
            unit: "".to_string(),
        };

        // add a sample log: f(t) = 2(t-3)^2 from 0 to 4
        let complex_times = Array1::<f64>::linspace(0., 6., 6001);
        let complex_log = ValueLog::<f64> {
            name: "complex".to_string(),
            time: complex_times.clone(),
            value: Array1::<f64>::from_iter(complex_times.iter().map(|t| 2. * (t - 3.).powi(2))),
            unit: "".to_string(),
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

    /// Test log_filter_times correctly gets the times from the log filters for an unbounded
    /// filter.
    #[test]
    fn test_log_filter_to_times_unbounded() {
        // note we test individual time ranges in data::sample_logs, so we only need to test
        // that this routine correctly maps over log names (and multiple bands for one log)
        let filters = Filters {
            time_filter_type: FilterType::Include,
            time_filters: Vec::<Filter>::new(),
            sample_log_filters: vec![
                LogFilter {
                    name: "a".to_string(),
                    log: "simple".to_string(),
                    lower: Some(2.),
                    upper: None,
                },
                LogFilter {
                    name: "b".to_string(),
                    log: "simple".to_string(),
                    lower: None,
                    upper: Some(3.),
                },
            ],
            amplitudes: HashMap::new(),
        };

        // logs from data::sample_logs tests
        let simple_times = Array1::<f64>::linspace(0., 4., 4001);
        let simple_log = ValueLog::<f64> {
            name: "simple".to_string(),
            time: simple_times.clone(),
            value: simple_times.clone(),
            unit: "".to_string(),
        };

        let mut logs = HashMap::<String, SampleLog>::new();
        logs.insert("simple".to_string(), SampleLog::F64(simple_log));

        let (starts, ends) = filters.get_log_filter_times(logs);
        let expected_starts = vec![2e9 as usize, 0];
        let expected_ends = vec![4e9 as usize, 3e9 as usize];
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
            .add_log_filter("name".to_string(), "temp".to_string(), Some(0.), Some(1.))
            .unwrap();
        assert_eq!(
            filters.sample_log_filters,
            vec![LogFilter {
                name: "name".to_string(),
                log: "temp".to_string(),
                lower: Some(0.),
                upper: Some(1.)
            }]
        )
    }

    /// Test adding multiple log filters.
    #[test]
    fn test_add_multiple_log_filter() {
        let mut filters = Filters::new();
        filters
            .add_log_filter("name".to_string(), "temp".to_string(), Some(0.), Some(1.))
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
                    lower: Some(0.),
                    upper: Some(1.)
                },
                LogFilter {
                    name: "name2".to_string(),
                    log: "p".to_string(),
                    lower: Some(5.),
                    upper: None
                },
                LogFilter {
                    name: "name3".to_string(),
                    log: "temp".to_string(),
                    lower: None,
                    upper: Some(8.)
                }
            ]
        )
    }

    /// Test an error is given when a log filter is given a duplicate name.
    #[test]
    fn test_add_log_filter_duplicate_name() {
        let mut filters = Filters::new();
        filters
            .add_log_filter(
                "filter1".to_string(),
                "temp".to_string(),
                Some(1.0),
                Some(2.0),
            )
            .unwrap();

        let result = filters.add_log_filter(
            "filter1".to_string(),
            "temp".to_string(),
            Some(3.0),
            Some(4.0),
        );
        assert!(result.is_err());
    }

    /// Test time filters can be removed correctly.
    #[test]
    fn test_remove_log_filter() {
        let mut filters = Filters::new();
        filters
            .add_log_filter(
                "filter1".to_string(),
                "temp".to_string(),
                Some(1.0),
                Some(2.0),
            )
            .unwrap();
        filters
            .add_log_filter(
                "filter2".to_string(),
                "pw".to_string(),
                Some(3.0),
                Some(4.0),
            )
            .unwrap();

        filters.remove_log_filter("filter1".to_string()).unwrap();

        assert_eq!(
            filters.sample_log_filters,
            vec![LogFilter {
                name: "filter2".to_string(),
                log: "pw".to_string(),
                lower: Some(3.),
                upper: Some(4.)
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
                lower: Some(5.),
                upper: Some(6.),
            }],
            amplitudes: HashMap::new(),
        };
        filters.amplitudes.insert(3, 5.);

        let mut temp_save_loc = temp_dir();
        temp_save_loc.push("filters_test_save_load.json");

        // these functions take strings but this is a PathBuf
        let tmp_path = temp_save_loc.to_string_lossy().to_string();

        let save_result = filters.save(tmp_path.clone());
        assert!(save_result.is_ok());
        let loaded_filters = Filters::load(tmp_path).unwrap();

        assert_eq!(filters, loaded_filters)
    }

    /// Test that saving a filter and loading it back in works for an unbounded log filter.
    #[test]
    fn test_save_load_unbounded_log() {
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
                lower: Some(5.),
                upper: None,
            }],
            amplitudes: HashMap::new(),
        };
        filters.amplitudes.insert(3, 5.);

        let mut temp_save_loc = temp_dir();
        temp_save_loc.push("filters_test_save_load_unbdd.json");

        // these functions take strings but this is a PathBuf
        let tmp_path = temp_save_loc.to_string_lossy().to_string();

        let save_result = filters.save(tmp_path.clone());
        assert!(save_result.is_ok());
        let loaded_filters = Filters::load(tmp_path).unwrap();

        assert_eq!(filters, loaded_filters)
    }

    /// Test loading a file that doesn't exist gives an error.
    #[test]
    fn test_load_nonexistent_file() {
        let filters = Filters::load("./fake_dir/fake_filters.json".to_string());
        assert!(filters.is_err())
    }

    /// Test `__repr__` reports the correct time filter type.
    #[test]
    fn test_repr_time_filter_type() {
        let mut filters = Filters::new();
        let repr = filters.__repr__();
        assert!(repr.starts_with("Time filter type: include"));
        filters.set_time_type("exclude".to_string()).unwrap();
        let repr = filters.__repr__();
        assert!(repr.starts_with("Time filter type: exclude"));
    }

    /// Test `__repr__` includes each time filter's name and bounds.
    #[test]
    fn test_repr_includes_time_filters() {
        let mut filters = Filters::new();
        filters
            .add_time_filter("filter1".to_string(), 1.0, 2.3)
            .unwrap();

        let repr = filters.__repr__();
        assert!(repr.contains("Time filters:"));
        assert!(repr.contains("filter1"));
        assert!(repr.contains('1'));
        assert!(repr.contains("2.3"));
    }

    /// Test `__repr__` includes each log filter's name, log, min and max.
    #[test]
    fn test_repr_includes_log_filters() {
        let mut filters = Filters::new();
        filters
            .add_log_filter(
                "logfilter".to_string(),
                "temp".to_string(),
                Some(1.0),
                Some(2.0),
            )
            .unwrap();

        let repr = filters.__repr__();
        assert!(repr.contains("Log filters:"));
        assert!(repr.contains("logfilter"));
        assert!(repr.contains("temp"));
        assert!(repr.contains('1'));
        assert!(repr.contains('2'));
    }

    /// Test `__repr__` labels an unbounded lower log filter with -inf.
    #[test]
    fn test_repr_log_filter_unbounded_lower() {
        let mut filters = Filters::new();
        filters
            .add_log_filter_below("logfilter".to_string(), "temp".to_string(), 5.0)
            .unwrap();

        let repr = filters.__repr__();
        assert!(repr.to_lowercase().contains("-inf"));
    }

    /// Test `__repr__` labels an unbounded upper log filter with inf.
    #[test]
    fn test_repr_log_filter_unbounded_upper() {
        let mut filters = Filters::new();
        filters
            .add_log_filter_above("logfilter".to_string(), "temp".to_string(), 5.0)
            .unwrap();

        let repr = filters.__repr__();
        assert!(repr.to_lowercase().contains("inf"));
    }

    /// Test `__repr__` shows "baseline" for the usize::MAX amplitude key
    /// instead of the raw number.
    #[test]
    fn test_repr_amplitude_baseline_label() {
        let mut filters = Filters::new();
        filters.set_amps_baseline(3.5);

        let repr = filters.__repr__();
        assert!(repr.contains("baseline"));
        assert!(repr.contains("3.5"));
    }

    /// Test `__repr__` shows the detector number for non-baseline amplitude filters.
    #[test]
    fn test_repr_amplitude_detector_label() {
        let mut filters = Filters::new();
        filters.set_amp(7, 2.2);

        let repr = filters.__repr__();
        assert!(repr.contains("7"));
        assert!(repr.contains("2.2"));
    }

    /// Test `__repr__` shows the detector number for non- and baseline amplitude filters.
    #[test]
    fn test_repr_amplitude_detector_mixed() {
        let mut filters = Filters::new();
        filters.set_amp(7, 2.2);
        filters.set_amps_baseline(1.5);

        let repr = filters.__repr__();
        assert!(repr.contains("7"));
        assert!(repr.contains("2.2"));
        assert!(repr.contains("baseline"));
        assert!(repr.contains("1.5"));
    }
}
