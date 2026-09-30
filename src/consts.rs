//! Constants and unit conversion helpers.

pub const S_TO_NS: f64 = 1e9;

pub const NS_TO_US: f32 = 1e-3;

pub trait ToNanoseconds {
    fn to_ns(&self) -> u64;
}

pub trait ToMicroseconds {
    fn to_micros(&self) -> f32;
}

impl ToMicroseconds for u32 {
    fn to_micros(&self) -> f32 {
        *self as f32 * NS_TO_US
    }
}

impl ToNanoseconds for f64 {
    fn to_ns(&self) -> u64 {
        (self * S_TO_NS) as u64
    }
}
