use std::str::FromStr;

use anyhow::{Error, Result};
use hdf5::types::{FloatSize, IntSize, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::Dataset;
use ndarray::Array1;

use crate::consts::S_TO_NS;
use crate::filters::LogPredicate;

// Pattern-matching is the only way to access the internal value log,
// so this macro lets you call a method of ValueLog from inside the
// SampleLog and return the new SampleLog
macro_rules! bind {
    ( $self_:expr, $method:ident ( $($arg:expr),* ) ) => {
        match $self_ {
            SampleLog::I8(log) => SampleLog::I8(log.$method($($arg),*)),
            SampleLog::I16(log) => SampleLog::I16(log.$method($($arg),*)),
            SampleLog::I32(log) => SampleLog::I32(log.$method($($arg),*)),
            SampleLog::I64(log) => SampleLog::I64(log.$method($($arg),*)),
            SampleLog::U8(log) => SampleLog::U8(log.$method($($arg),*)),
            SampleLog::U16(log) => SampleLog::U16(log.$method($($arg),*)),
            SampleLog::U32(log) => SampleLog::U32(log.$method($($arg),*)),
            SampleLog::U64(log) => SampleLog::U64(log.$method($($arg),*)),
            SampleLog::F32(log) => SampleLog::F32(log.$method($($arg),*)),
            SampleLog::F64(log) => SampleLog::F64(log.$method($($arg),*)),
            SampleLog::Str(log) => SampleLog::Str(log.$method($($arg),*)),
        }
    }
}

#[derive(PartialEq)]
pub enum SampleLog {
    I8(ValueLog<i8>),
    I16(ValueLog<i16>),
    I32(ValueLog<i32>),
    I64(ValueLog<i64>),
    U8(ValueLog<u8>),
    U16(ValueLog<u16>),
    U32(ValueLog<u32>),
    U64(ValueLog<u64>),
    F32(ValueLog<f32>),
    F64(ValueLog<f64>),
    // string logs are always stored as VarLenUnicode
    Str(ValueLog<VarLenUnicode>),
}

/// Read the `units` attribute of a value dataset, defaulting to no units.
fn read_unit(value: &Dataset) -> String {
    match value.attr("units") {
        Ok(units) => units
            .read_scalar::<VarLenUnicode>()
            .map(|u| u.to_string())
            .unwrap_or_default(),
        Err(_) => String::new(),
    }
}

impl SampleLog {
    /// Create a new SampleLog.
    pub fn new(log_name: &String, time: Array1<f64>, value: Dataset) -> Result<SampleLog> {
        let dtype = value.dtype()?.to_descriptor()?;
        let unit = read_unit(&value);

        // String logs are handled before the numeric macro because both variable-length
        // encodings normalise onto a single Str variant, so they do not fit the
        // one-HDF5-type-per-Rust-type shape the macro assumes.
        //
        // Fixed-length strings are deliberately not supported: FixedAscii<N>/FixedUnicode<N>
        // carry N as a const generic while TypeDescriptor::FixedAscii(usize) carries it as a
        // runtime value, so dispatching between them needs a hardcoded ladder of sizes.
        // ISIS writes variable-length UTF-8, so this buys nothing.
        match dtype {
            TypeDescriptor::VarLenUnicode => {
                return Ok(SampleLog::Str(ValueLog::<VarLenUnicode> {
                    name: log_name.clone(),
                    time,
                    value: value.read_1d()?,
                    unit,
                }))
            }
            TypeDescriptor::VarLenAscii => {
                let ascii: Array1<VarLenAscii> = value.read_1d()?;
                let values = ascii
                    .iter()
                    .map(|s| VarLenUnicode::from_str(s))
                    .collect::<Result<Vec<VarLenUnicode>, _>>()?;
                return Ok(SampleLog::Str(ValueLog::<VarLenUnicode> {
                    name: log_name.clone(),
                    time,
                    value: values.into(),
                    unit,
                }));
            }
            _ => {}
        }

        // this macro just lets us write the type conversions,
        // and it handles the actual construction which is always the same
        // e.g. TypeDescriptor::Integer(IntSize::U1) => I8: i8 is converted to the branch
        //
        //    TypeDescriptor::Integer(IntSize::U1) => SampleLog::I8(ValueLog<i8> {
        //        time: time,
        //        value: value.read_1d()?
        //    }
        //
        macro_rules! make_value_log {
            ( $( $hdf5_type:pat => $variant:ident : $type:ty ),+ $(,)? ) => {
                match dtype {
                    $(
                        $hdf5_type => {
                            SampleLog::$variant(ValueLog::<$type> {
                            name: log_name.clone(),
                            time,
                            value: value.read_1d()?,
                            unit
                            })
                        },
                    )+
                    other_type => return Err(Error::msg(format!(
                            "Sample log type {other_type} for log {log_name} is not supported.
                            Supported types are Integer, Float and variable-length String.",
                    )))
                }
            };
        }
        let log = make_value_log! {
            TypeDescriptor::Integer(IntSize::U1) => I8: i8,
            TypeDescriptor::Integer(IntSize::U2) => I16: i16,
            TypeDescriptor::Integer(IntSize::U4) => I32: i32,
            TypeDescriptor::Integer(IntSize::U8) => I64: i64,
            TypeDescriptor::Unsigned(IntSize::U1) => U8: u8,
            TypeDescriptor::Unsigned(IntSize::U2) => U16: u16,
            TypeDescriptor::Unsigned(IntSize::U4) => U32: u32,
            TypeDescriptor::Unsigned(IntSize::U8) => U64: u64,
            TypeDescriptor::Float(FloatSize::U4) => F32: f32,
            TypeDescriptor::Float(FloatSize::U8) => F64: f64,
        };
        Ok(log)
    }

    /// The name of the log.
    pub fn name(&self) -> &str {
        match self {
            SampleLog::I8(log) => &log.name,
            SampleLog::I16(log) => &log.name,
            SampleLog::I32(log) => &log.name,
            SampleLog::I64(log) => &log.name,
            SampleLog::U8(log) => &log.name,
            SampleLog::U16(log) => &log.name,
            SampleLog::U32(log) => &log.name,
            SampleLog::U64(log) => &log.name,
            SampleLog::F32(log) => &log.name,
            SampleLog::F64(log) => &log.name,
            SampleLog::Str(log) => &log.name,
        }
    }

    /// The distinct values a string log takes, in order of first appearance.
    pub fn distinct_values(&self) -> Option<Vec<String>> {
        match self {
            SampleLog::Str(log) => {
                let mut seen = Vec::<String>::new();
                for value in log.value.iter() {
                    let value = value.to_string();
                    if !seen.contains(&value) {
                        seen.push(value);
                    }
                }
                Some(seen)
            }
            _ => None,
        }
    }

    /// Given a filter predicate, get the list of time starts and ends
    /// corresponding to the log filter.
    ///
    /// Returns an error if the predicate does not suit the log's type, e.g. a range
    /// filter applied to a log holding text
    pub fn to_time_ranges(&self, predicate: &LogPredicate) -> Result<(Vec<usize>, Vec<usize>)> {
        // this macro pairs every numeric variant with every predicate, so that the
        // match stays exhaustive without a catch-all arm hiding a future variant
        macro_rules! ranges_for_predicate {
            ( $( $variant:ident : $type:ty ),+ $(,)? ) => {
                match (self, predicate) {
                    (SampleLog::Str(log), LogPredicate::Equals(value)) => {
                        Ok(log.to_time_ranges_eq(value))
                    }
                    (SampleLog::Str(log), LogPredicate::Range { .. }) => Err(Error::msg(format!(
                        "Sample log {} holds text, so it cannot be filtered on a range of \
                         values. Use add_string_log_filter to filter it on a specific value.",
                        log.name,
                    ))),
                    $(
                        (SampleLog::$variant(log), LogPredicate::Range { lower, upper }) => {
                            Ok(log.to_time_ranges(
                                &(lower.unwrap_or(-f64::INFINITY) as $type),
                                &(upper.unwrap_or(f64::INFINITY) as $type),
                            ))
                        },
                        (SampleLog::$variant(log), LogPredicate::Equals(_)) => {
                            Err(Error::msg(format!(
                                "Sample log {} holds numbers, so it cannot be filtered on a \
                                 string value. Use add_log_filter, add_log_filter_above or \
                                 add_log_filter_below.",
                                log.name,
                            )))
                        },
                    )+
                }
            };
        }
        ranges_for_predicate! {
            I8: i8, I16: i16, I32: i32, I64: i64,
            U8: u8, U16: u16, U32: u32, U64: u64,
            F32: f32, F64: f64,
        }
    }

    /// Given a list of filter starts and ends, return the sample log with filter applied.
    pub fn apply_filters(
        &self,
        start_times: &[usize],
        end_times: &[usize],
        include: bool,
    ) -> SampleLog {
        bind! {self, apply_filters(start_times, end_times, include)}
    }
}

#[derive(Clone, PartialEq)]
pub struct ValueLog<T> {
    pub name: String,
    pub time: Array1<f64>,
    pub value: Array1<T>,
    pub unit: String,
}

impl<T> ValueLog<T>
where
    T: PartialOrd,
{
    /// Internal implementation of SampleLog.to_time_ranges
    fn to_time_ranges(&self, lower: &T, upper: &T) -> (Vec<usize>, Vec<usize>) {
        let mut starts = Vec::<usize>::new();
        let mut ends = Vec::<usize>::new();
        let mut in_range: bool = false; // represents whether we are currently in the band
        for (index, value) in self.value.iter().enumerate() {
            if !in_range {
                if value <= upper && value >= lower {
                    starts.push((self.time[index] * S_TO_NS) as usize);
                    in_range = true;
                }
            } else {
                if value > upper || value < lower {
                    ends.push((self.time[index - 1] * S_TO_NS) as usize);
                    in_range = false;
                }
            }
        }

        // if still in range at end, we need to add the last datapoint to the ends
        if in_range {
            ends.push((self.time.last().unwrap() * S_TO_NS) as usize);
        }
        (starts, ends)
    }
}

impl ValueLog<VarLenUnicode> {
    /// Internal implementation of SampleLog.to_time_ranges for a string log.
    fn to_time_ranges_eq(&self, value: &str) -> (Vec<usize>, Vec<usize>) {
        let mut starts = Vec::<usize>::new();
        let mut ends = Vec::<usize>::new();
        let mut in_range: bool = false; // whether we are currently in a matching run
                                        // fold the needle once, rather than once per sample
        let needle = value.to_lowercase();

        for (index, entry) in self.value.iter().enumerate() {
            let matches = entry.to_lowercase() == needle;
            if !in_range {
                if matches {
                    starts.push((self.time[index] * S_TO_NS) as usize);
                    in_range = true;
                }
            } else if !matches {
                // the run ends where the next value takes over, not at the previous sample
                ends.push((self.time[index] * S_TO_NS) as usize);
                in_range = false;
            }
        }

        // if still matching at the end, the run ends at the last datapoint. Note this is
        // the last logged time, not the end of the run - a log that stops being written
        // early will truncate the filter here.
        if in_range {
            ends.push((self.time.last().unwrap() * S_TO_NS) as usize);
        }
        (starts, ends)
    }
}

impl<T> ValueLog<T>
where
    T: Clone,
{
    /// Internal implementation of SampleLog.apply_filters.
    fn apply_filters(
        &self,
        start_times: &[usize],
        end_times: &[usize],
        include: bool,
    ) -> ValueLog<T> {
        // Sort filters to make overlaps easier to handle
        let mut ranges: Vec<(usize, usize)> = start_times
            .iter()
            .copied()
            .zip(end_times.iter().copied())
            .collect();
        ranges.sort_unstable();

        let mut new_times = Vec::<f64>::with_capacity(self.time.len());
        let mut new_values = Vec::<T>::with_capacity(self.value.len());
        let mut next_range = 0;
        let mut reach: Option<usize> = None;

        for (k, t) in self.time.iter().enumerate() {
            let time_ns = (t * S_TO_NS) as usize;
            while next_range < ranges.len() && ranges[next_range].0 <= time_ns {
                let end = ranges[next_range].1;
                reach = Some(reach.map_or(end, |r: usize| r.max(end)));
                next_range += 1;
            }
            // ranges are closed at both ends, matching to_time_ranges
            let in_range = reach.is_some_and(|end| time_ns <= end);
            if in_range == include {
                new_times.push(*t);
                new_values.push(self.value[k].clone());
            }
        }
        ValueLog::<T> {
            name: self.name.clone(),
            time: new_times.into(),
            value: new_values.into(),
            unit: self.unit.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a string value log from (time, value) pairs.
    fn string_log(entries: &[(f64, &str)]) -> ValueLog<VarLenUnicode> {
        ValueLog::<VarLenUnicode> {
            name: "status".to_string(),
            time: entries.iter().map(|(t, _)| *t).collect::<Vec<f64>>().into(),
            value: entries
                .iter()
                .map(|(_, v)| VarLenUnicode::from_str(v).unwrap())
                .collect::<Vec<VarLenUnicode>>()
                .into(),
            unit: "".to_string(),
        }
    }

    fn ns(seconds: f64) -> usize {
        (seconds * S_TO_NS) as usize
    }

    #[test]
    fn test_str_time_ranges() {
        let log = string_log(&[(0., "IDLE"), (3., "RUNNING"), (10., "PAUSED")]);

        let (starts, ends) = log.to_time_ranges_eq("RUNNING");

        assert_eq!(starts, vec![ns(3.)]);
        assert_eq!(ends, vec![ns(10.)]);
    }

    /// A run still matching at the end of the log ends at the last logged time.
    #[test]
    fn test_str_time_ranges_at_end() {
        let log = string_log(&[(0., "IDLE"), (3., "RUNNING"), (10., "RUNNING")]);

        let (starts, ends) = log.to_time_ranges_eq("RUNNING");

        assert_eq!(starts, vec![ns(3.)]);
        assert_eq!(ends, vec![ns(10.)]);
    }

    /// A log whose first value already matches starts at its first timestamp.
    #[test]
    fn test_str_time_ranges_at_start() {
        let log = string_log(&[(2., "RUNNING"), (5., "PAUSED")]);

        let (starts, ends) = log.to_time_ranges_eq("RUNNING");

        assert_eq!(starts, vec![ns(2.)]);
        assert_eq!(ends, vec![ns(5.)]);
    }

    /// Alternating values produce one range per matching run.
    #[test]
    fn test_str_time_ranges_multiple_runs() {
        let log = string_log(&[
            (0., "RUNNING"),
            (1., "PAUSED"),
            (2., "RUNNING"),
            (3., "RUNNING"),
            (4., "IDLE"),
        ]);

        let (starts, ends) = log.to_time_ranges_eq("RUNNING");

        assert_eq!(starts, vec![ns(0.), ns(2.)]);
        assert_eq!(ends, vec![ns(1.), ns(4.)]);
    }

    /// A value that never appears produces no ranges.
    #[test]
    fn test_str_time_ranges_no_match() {
        let log = string_log(&[(0., "IDLE"), (3., "PAUSED")]);

        let (starts, ends) = log.to_time_ranges_eq("RUNNING");

        assert!(starts.is_empty());
        assert!(ends.is_empty());
    }

    /// Matching ignores case, in both the log value and the requested value.
    #[test]
    fn test_time_ranges_eq_is_case_insensitive() {
        let log = string_log(&[(0., "idle"), (3., "Running"), (10., "PAUSED")]);

        let (starts, ends) = log.to_time_ranges_eq("rUnNiNg");

        assert_eq!(starts, vec![ns(3.)]);
        assert_eq!(ends, vec![ns(10.)]);
    }

    /// A string log cannot be filtered on a numeric range.
    #[test]
    fn test_range_predicate_on_string_log_errors() {
        let log = SampleLog::Str(string_log(&[(0., "IDLE")]));

        let result = log.to_time_ranges(&LogPredicate::Range {
            lower: Some(1.),
            upper: Some(2.),
        });

        let error = result.unwrap_err().to_string();
        assert!(error.contains("holds text"), "unexpected error: {error}");
        assert!(error.contains("add_string_log_filter"));
    }

    /// A numeric log cannot be filtered on a string value.
    #[test]
    fn test_equals_predicate_on_numeric_log_errors() {
        let log = SampleLog::F64(ValueLog::<f64> {
            name: "temp".to_string(),
            time: Array1::<f64>::linspace(0., 4., 5),
            value: Array1::<f64>::linspace(0., 4., 5),
            unit: "".to_string(),
        });

        let result = log.to_time_ranges(&LogPredicate::Equals("RUNNING".to_string()));

        let error = result.unwrap_err().to_string();
        assert!(error.contains("holds numbers"), "unexpected error: {error}");
    }

    /// distinct_values lists a string log's values in order of first appearance.
    #[test]
    fn test_distinct_values() {
        let log = SampleLog::Str(string_log(&[
            (0., "IDLE"),
            (1., "RUNNING"),
            (2., "RUNNING"),
            (3., "IDLE"),
            (4., "PAUSED"),
        ]));

        assert_eq!(
            log.distinct_values(),
            Some(vec![
                "IDLE".to_string(),
                "RUNNING".to_string(),
                "PAUSED".to_string()
            ])
        );
    }

    /// Test apply_filters works on a string log.
    #[test]
    fn test_apply_filters_string_log() {
        let log = SampleLog::Str(string_log(&[
            (0., "IDLE"),
            (1., "RUNNING"),
            (2., "PAUSED"),
            (3., "IDLE"),
        ]));

        let filtered = log.apply_filters(&[ns(1.)], &[ns(2.)], true);

        match filtered {
            SampleLog::Str(log) => {
                let values: Vec<String> = log.value.iter().map(|v| v.to_string()).collect();
                assert_eq!(values, vec!["RUNNING".to_string(), "PAUSED".to_string()]);
            }
            _ => panic!("filtering a string log should return a string log"),
        }
    }

    /// Test log_filter_times correctly gets the times from the log filters for a simple case.
    #[test]
    fn test_time_ranges_simple() {
        // add a sample log: f(t) = t from 0 to 4
        let times = Array1::<f64>::linspace(0., 4., 4001);
        let value_log = ValueLog::<f64> {
            name: "temp".to_string(),
            time: times.clone(),
            value: times.clone(),
            unit: "".to_string(),
        };

        let (starts, ends) = value_log.to_time_ranges(&1., &2.);
        let expected_starts = vec![(1. * S_TO_NS) as usize];
        let expected_ends = vec![(2. * S_TO_NS) as usize];

        assert_eq!(starts, expected_starts);
        assert_eq!(ends, expected_ends)
    }

    /// Test log_filter_times correctly gets the times from the log filters when the start is
    /// included.
    #[test]
    fn test_time_ranges_with_start() {
        // add a sample log: f(t) = t from 0 to 4
        let times = Array1::<f64>::linspace(0., 4., 4001);
        let value_log = ValueLog::<f64> {
            name: "temp".to_string(),
            time: times.clone(),
            value: times.clone(),
            unit: "".to_string(),
        };

        let (starts, ends) = value_log.to_time_ranges(&0., &2.);
        let expected_starts = vec![0];
        let expected_ends = vec![(2. * S_TO_NS) as usize];

        assert_eq!(starts, expected_starts);
        assert_eq!(ends, expected_ends)
    }

    /// Test log_filter_times correctly gets the times from the log filters when the end is
    /// included.
    #[test]
    fn test_time_ranges_with_end() {
        // add a sample log: f(t) = t from 0 to 4
        let times = Array1::<f64>::linspace(0., 4., 4001);
        let value_log = ValueLog::<f64> {
            name: "temp".to_string(),
            time: times.clone(),
            value: times.clone(),
            unit: "".to_string(),
        };

        let (starts, ends) = value_log.to_time_ranges(&1., &4.);
        let expected_starts = vec![(1. * S_TO_NS) as usize];
        let expected_ends = vec![(4. * S_TO_NS) as usize];

        assert_eq!(starts, expected_starts);
        assert_eq!(ends, expected_ends)
    }

    /// Test log_filter_times correctly gets the times from the log filters for a case with
    /// multiple ranges.
    #[test]
    fn test_time_ranges_complex() {
        // add a sample log: f(t) = 2(t-3)^2 from 0 to 4
        let times = Array1::<f64>::linspace(0., 6., 6001);
        let value_log = ValueLog::<f64> {
            name: "temp".to_string(),
            time: times.clone(),
            value: Array1::<f64>::from_iter(times.iter().map(|t| 2. * (t - 3.).powi(2))),
            unit: "".to_string(),
        };

        // f(t) is between 2 and 8 for t = 1-2 and 4-5
        let (starts, ends) = value_log.to_time_ranges(&2., &8.);
        let expected_starts = vec![(1. * S_TO_NS) as usize, (4. * S_TO_NS) as usize];
        let expected_ends = vec![(2. * S_TO_NS) as usize, (5. * S_TO_NS) as usize];

        assert_eq!(starts, expected_starts);
        assert_eq!(ends, expected_ends)
    }

    /// A sample ValueLog for testing.
    fn sample_log() -> ValueLog<f64> {
        let name = "temp".to_string();
        let time = Array1::<f64>::from_vec(vec![0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
        let value = Array1::<f64>::from_vec(vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);

        ValueLog::<f64> {
            name,
            time,
            value,
            unit: "".to_string(),
        }
    }

    /// Test an include filter keeps exactly the values inside its ranges.
    #[test]
    fn test_apply_filters_include() {
        let log = sample_log();

        // filters are [0.15, 0.22], [0.55, 0.83]
        // 0  1  2  3  4  5  6  7  8  9   values
        //     ^--^        ^--------^     include
        let filter_starts = vec![(0.15 * S_TO_NS) as usize, (0.55 * S_TO_NS) as usize];
        let filter_ends = vec![(0.22 * S_TO_NS) as usize, (0.83 * S_TO_NS) as usize];
        let new_log = log.apply_filters(&filter_starts, &filter_ends, true);

        let expected_vals = Array1::<f64>::from_vec(vec![2., 6., 7., 8.]);
        assert_eq!(new_log.value, expected_vals)
    }

    /// Test applying filters successfully 'flattens' filtered-out values.
    #[test]
    fn test_apply_filters() {
        let log = sample_log();

        // filters are [0.15, 0.22], [0.55, 0.83]
        // 0  1  2  3  4  5  6  7  8  9   values
        //     ^--^        ^--------^     exclude
        let filter_starts = vec![(0.15 * S_TO_NS) as usize, (0.55 * S_TO_NS) as usize];
        let filter_ends = vec![(0.22 * S_TO_NS) as usize, (0.83 * S_TO_NS) as usize];
        let new_log = log.apply_filters(&filter_starts, &filter_ends, false);

        let expected_vals = Array1::<f64>::from_vec(vec![0., 1., 3., 4., 5., 9.]);
        assert_eq!(new_log.value, expected_vals)
    }

    /// Test applying filters works when the filters have an overlap.
    #[test]
    fn test_apply_filters_overlap() {
        let log = sample_log();

        // filters are [0.06, 0.35], [0.22, 0.56], [0.71, 0.89]
        // 0  1  2  3  4  5  6  7  8  9   values
        //   ^-------^           ^---^    exclude
        //        ^--------^
        let filter_starts = vec![
            (0.06 * S_TO_NS) as usize,
            (0.22 * S_TO_NS) as usize,
            (0.71 * S_TO_NS) as usize,
        ];
        let filter_ends = vec![
            (0.35 * S_TO_NS) as usize,
            (0.56 * S_TO_NS) as usize,
            (0.89 * S_TO_NS) as usize,
        ];
        let new_log = log.apply_filters(&filter_starts, &filter_ends, false);

        let expected_vals = Array1::<f64>::from_vec(vec![0., 6., 7., 9.]);
        assert_eq!(new_log.value, expected_vals)
    }

    /// An include filter over overlapping ranges keeps the union of them.
    #[test]
    fn test_apply_filters_overlap_include() {
        let log = sample_log();

        // filters are [0.06, 0.35], [0.22, 0.56], [0.71, 0.89]
        // 0  1  2  3  4  5  6  7  8  9   values
        //   ^-------^           ^---^    include
        //        ^--------^
        let filter_starts = vec![
            (0.06 * S_TO_NS) as usize,
            (0.22 * S_TO_NS) as usize,
            (0.71 * S_TO_NS) as usize,
        ];
        let filter_ends = vec![
            (0.35 * S_TO_NS) as usize,
            (0.56 * S_TO_NS) as usize,
            (0.89 * S_TO_NS) as usize,
        ];
        let new_log = log.apply_filters(&filter_starts, &filter_ends, true);

        let expected_vals = Array1::<f64>::from_vec(vec![1., 2., 3., 4., 5., 8.]);
        assert_eq!(new_log.value, expected_vals)
    }

    /// Ranges arrive concatenated per filter, so they are not sorted; the result must
    /// not depend on the order they are given in.
    #[test]
    fn test_apply_filters_unsorted_ranges() {
        let log = sample_log();

        let filter_starts = vec![(0.55 * S_TO_NS) as usize, (0.15 * S_TO_NS) as usize];
        let filter_ends = vec![(0.83 * S_TO_NS) as usize, (0.22 * S_TO_NS) as usize];
        let new_log = log.apply_filters(&filter_starts, &filter_ends, true);

        let expected_vals = Array1::<f64>::from_vec(vec![2., 6., 7., 8.]);
        assert_eq!(new_log.value, expected_vals)
    }

    /// Test applying filters works when the first value is included in a filter.
    #[test]
    fn test_apply_filters_including_start() {
        let log = sample_log();

        // filter [0, 0.61]
        // 0  1  2  3  4  5  6  7  8  9  values
        // ^----------------^          exclude
        let filter_starts = vec![0];
        let filter_ends = vec![(0.61 * S_TO_NS) as usize];
        let new_log = log.apply_filters(&filter_starts, &filter_ends, false);

        let expected_vals = Array1::<f64>::from_vec(vec![7., 8., 9.]);
        assert_eq!(new_log.value, expected_vals)
    }

    #[test]
    fn test_apply_filters_including_end() {
        let log = sample_log();

        // filter [0.45, inf]
        // 0  1  2  3  4  5  6  7  8  9  values
        //              ^------------... exclude
        let filter_starts = vec![(0.45 * S_TO_NS) as usize];
        let filter_ends = vec![usize::MAX];
        let new_log = log.apply_filters(&filter_starts, &filter_ends, false);

        let expected_vals = Array1::<f64>::from_vec(vec![0., 1., 2., 3., 4.]);
        assert_eq!(new_log.value, expected_vals)
    }
}
