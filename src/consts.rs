//! Constants.

pub const S_TO_NS: f64 = 1e9;

pub trait ToNanoseconds {
    fn to_ns(&self) -> usize;
}

impl ToNanoseconds for f64 {
    fn to_ns(&self) -> usize {
        (self * S_TO_NS) as usize
    }
}
