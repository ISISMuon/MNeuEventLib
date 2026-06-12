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
        let in_range: bool = false; // represents whether we are currently in the band
        for (index, value) in self.value.iter().enumerate() {
            if !in_range {
                if value <= upper && value >= lower {
                    starts.push((self.time[index] * S_TO_NS) as usize);
                }
            } else {
                if value > upper || value < lower {
                    ends.push((self.time[index] * S_TO_NS) as usize);
                }
            }
        }
        (starts, ends)
    }
}
