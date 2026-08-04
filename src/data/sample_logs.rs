use anyhow::{Error, Result};
use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use hdf5::Dataset;
use ndarray::Array1;

use crate::consts::S_TO_NS;

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
}

impl SampleLog {
    /// Create a new SampleLog.
    pub fn new(log_name: &String, time: Array1<f64>, value: Dataset) -> Result<SampleLog> {
        let dtype = value.dtype()?.to_descriptor()?;

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
                        $hdf5_type => SampleLog::$variant(ValueLog::<$type> {
                            name: log_name.clone(),
                            time,
                            value: value.read_1d()?,
                        }),
                    )+
                    other_type => return Err(Error::msg(format!(
                            "Sample log type {other_type} for log {log_name} is not supported.
                            Supported types are Integer and Float.",
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

    /// Given a lower and upper limit, get the list of time starts and ends
    /// corresponding to the log filter.
    pub fn to_time_ranges(&self, lower: f64, upper: f64) -> (Vec<usize>, Vec<usize>) {
        match self {
            SampleLog::I8(log) => log.to_time_ranges(&(lower as i8), &(upper as i8)),
            SampleLog::I16(log) => log.to_time_ranges(&(lower as i16), &(upper as i16)),
            SampleLog::I32(log) => log.to_time_ranges(&(lower as i32), &(upper as i32)),
            SampleLog::I64(log) => log.to_time_ranges(&(lower as i64), &(upper as i64)),
            SampleLog::U8(log) => log.to_time_ranges(&(lower as u8), &(upper as u8)),
            SampleLog::U16(log) => log.to_time_ranges(&(lower as u16), &(upper as u16)),
            SampleLog::U32(log) => log.to_time_ranges(&(lower as u32), &(upper as u32)),
            SampleLog::U64(log) => log.to_time_ranges(&(lower as u64), &(upper as u64)),
            SampleLog::F32(log) => log.to_time_ranges(&(lower as f32), &(upper as f32)),
            SampleLog::F64(log) => log.to_time_ranges(&lower, &upper),
        }
    }

    /// Given a list of filter starts and ends, return the sample log with filter applied.
    ///
    /// Assumes that start_times and end_times are sorted arrays.
    pub fn apply_filters(&self, start_times: &Vec<usize>, end_times: &Vec<usize>) -> SampleLog {
        // we need to use pattern matching to access the inside of each log, but what we're
        // doing is essentially just
        // sample_log(log) -> sample_log(log.apply_filters())
        macro_rules! apply_filters_for_type {
            ( $($type:path),+ ) => {
                match self {
                    $(
                        $type(log) => $type(log.apply_filters(start_times, end_times)),
                    )+
                }
            }
        }
        apply_filters_for_type! {
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
    }
}

#[derive(Clone)]
pub struct ValueLog<T> {
    pub name: String,
    pub time: Array1<f64>,
    pub value: Array1<T>,
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

impl<T> ValueLog<T>
where
    T: Clone,
{
    /// Internal implementation of SampleLog.apply_filters.
    fn apply_filters(&self, start_times: &Vec<usize>, end_times: &Vec<usize>) -> ValueLog<T> {
        // we use these indices to ignore overlaps in filters.
        //
        // Note that as the start_times and end_times are sorted, we will never get a scenario
        // where one interval lies completely inside another:
        //
        // let (--) be one filter and [~~] another,
        // then the filter pair [s1, e1], [s2, e2] like so
        // s1      e1
        // (-------)
        //   [~~~]
        //   s2  e2
        // has starts [s1, s2], ends [e1, e2]
        // which would be sorted as starts [s1, s2], ends [e2, e1]
        // which parses as filters [s1, e2], [s2, e1]
        // s1    e2
        // (-----)
        //   [~~~~~]
        //   s2    e1
        //
        // This means we can ignore overlaps by checking if we've passed an extra start value,
        // and if so, just skipping an end value to compensate
        let mut current_start_idx = 0;
        let mut current_end_idx = 0;
        let mut in_filter = false;
        let mut new_times = Vec::<f64>::with_capacity(self.time.len());
        let mut new_values = Vec::<T>::with_capacity(self.value.len());
        let max_start_idx = start_times.len() - 1;

        for (k, t) in self.time.iter().enumerate() {
            let time_ns = (t * S_TO_NS) as usize;
            if in_filter {
                if time_ns >= end_times[current_end_idx] {
                    // end of interval
                    in_filter = false;
                    current_end_idx += 1;
                    current_start_idx += 1;
                    new_times.push(*t);
                    new_values.push(self.value[k].clone());
                } else if current_start_idx < max_start_idx
                    && time_ns >= start_times[current_start_idx + 1]
                {
                    // overlap detected
                    current_start_idx += 1;
                    current_end_idx += 1
                }
            } else if current_start_idx < start_times.len()
                && time_ns >= start_times[current_start_idx]
            {
                // start of interval
                in_filter = true;
            } else {
                // not in filter; append to new log
                new_times.push(*t);
                new_values.push(self.value[k].clone());
            }
        }
        ValueLog::<T> {
            name: self.name.clone(),
            time: new_times.into(),
            value: new_values.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test log_filter_times correctly gets the times from the log filters for a simple case.
    #[test]
    fn test_time_ranges_simple() {
        // add a sample log: f(t) = t from 0 to 4
        let times = Array1::<f64>::linspace(0., 4., 4001);
        let value_log = ValueLog::<f64> {
            name: "temp".to_string(),
            time: times.clone(),
            value: times.clone(),
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

        ValueLog::<f64> { name, time, value }
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
        let new_log = log.apply_filters(&filter_starts, &filter_ends);

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
        let new_log = log.apply_filters(&filter_starts, &filter_ends);

        let expected_vals = Array1::<f64>::from_vec(vec![0., 6., 7., 9.]);
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
        let new_log = log.apply_filters(&filter_starts, &filter_ends);

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
        let new_log = log.apply_filters(&filter_starts, &filter_ends);

        let expected_vals = Array1::<f64>::from_vec(vec![0., 1., 2., 3., 4.]);
        assert_eq!(new_log.value, expected_vals)
    }
}
