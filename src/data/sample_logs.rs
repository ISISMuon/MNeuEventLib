use ndarray::Array1;

use crate::consts::S_TO_NS;

#[allow(dead_code)] // to be implemented by log filters
pub enum SampleLog {
    Int(ValueLog<i32>),
    Float(ValueLog<f64>),
}

impl SampleLog {
    /// Given a lower and upper limit, get the list of time starts and ends
    /// corresponding to the log filter.
    pub fn to_time_ranges(&self, lower: f64, upper: f64) -> (Vec<usize>, Vec<usize>) {
        match self {
            SampleLog::Int(log) => log.to_time_ranges(&(lower as i32), &(upper as i32)),
            SampleLog::Float(log) => log.to_time_ranges(&lower, &upper),
        }
    }
}

#[allow(dead_code)] // to be implemented by log filters
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
