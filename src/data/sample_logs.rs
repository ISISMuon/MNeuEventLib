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
}

pub struct ValueLog<T> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Test log_filter_times correctly gets the times from the log filters for a complex case.
    #[test]
    fn test_time_ranges_simple() {
        // add a sample log: f(t) = t from 0 to 4
        let times = Array1::<f64>::linspace(0., 4., 4001);
        let value_log = ValueLog::<f64> {
            time: times.clone(),
            value: times.clone(),
        };

        let (starts, ends) = value_log.to_time_ranges(&1., &2.);
        let expected_starts = vec![1e9 as usize];
        let expected_ends = vec![2e9 as usize];

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
            time: times.clone(),
            value: Array1::<f64>::from_iter(times.iter().map(|t| 2. * (t - 3.).powi(2))),
        };

        // f(t) is between 2 and 8 for t = 1-2 and 4-5
        let (starts, ends) = value_log.to_time_ranges(&2., &8.);
        let expected_starts = vec![1e9 as usize, 4e9 as usize];
        let expected_ends = vec![2e9 as usize, 5e9 as usize];

        assert_eq!(starts, expected_starts);
        assert_eq!(ends, expected_ends)
    }
}
