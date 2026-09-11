//! Constants and unit conversion helpers.

pub const S_TO_NS: f64 = 1e9;

pub trait ToNanoseconds {
    fn to_ns(&self) -> u64;
}

impl ToNanoseconds for f64 {
    fn to_ns(&self) -> u64 {
        (self * S_TO_NS) as u64
    }
}
