//! Constants and unit conversion helpers.

pub const S_TO_NS: f64 = 1e9;

pub const NS_TO_US: f32 = 1e-3;

/// Default histogram minimum time (microseconds).
pub const DEFAULT_HIST_MIN_TIME: u32 = 0;

/// Default histogram maximum time (microseconds).
pub const DEFAULT_HIST_MAX_TIME: u32 = 32768;

/// Default number of histogram bins.
pub const DEFAULT_HIST_N_BINS: usize = 2048;

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
