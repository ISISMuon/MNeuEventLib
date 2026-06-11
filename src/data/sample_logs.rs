use ndarray::Array1;

pub enum SampleLog {
    Bool(ValueLog<bool>),
    Int(ValueLog<i32>),
    Float(ValueLog<f64>),
}

pub struct ValueLog<T> {
    pub time: Array1<f64>,
    pub value: Array1<T>,
}
