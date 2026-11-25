use ndarray::Array1;

use crate::utils::binary_search;

/// Fast, zero-allocation binary search lookup returning `(lower_frame, upper_frame)`
/// containing all events in the half-open event index range `[lower_event, upper_event)`.
#[inline]
pub fn frame_range(
    start_index: &Array1<usize>,
    lower_event: usize,
    upper_event: usize,
) -> (usize, usize) {
    let n_frames = start_index.len();
    let lower_frame_index = binary_search::<usize>(start_index, 0, n_frames, lower_event);
    let upper_frame_index = binary_search::<usize>(start_index, 0, n_frames, upper_event);
    (lower_frame_index, upper_frame_index)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test frame range returns correct lower and upper frame indices.
    #[test]
    fn test_frame_range() {
        let starts = Array1::from_vec(vec![0, 5, 11, 16]);
        assert_eq!(frame_range(&starts, 7, 15), (1, 2));
    }
}
