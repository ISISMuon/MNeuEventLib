use ndarray::Array1;

use crate::consts::S_TO_NS;

#[allow(dead_code)] // to be implemented by log filters
pub enum SampleLog {
    Bool(ValueLog<bool>),
    Int(ValueLog<i32>),
    Float(ValueLog<f64>),
}

#[allow(dead_code)] // to be implemented by log filters
pub struct ValueLog<T> {
    pub time: Array1<f64>,
    pub value: Array1<T>,
}

impl<T> ValueLog<T>
where
    T: Ord,
{
    /// Given a lower and upper limit, get the list of time starts and ends 
    /// corresponding to the log filter.
    pub fn to_time_ranges(&self, lower: &T, upper: &T) -> (Vec<usize>, Vec<usize>) {
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
