use ndarray::Array1;

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
