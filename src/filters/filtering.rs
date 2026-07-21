use ndarray::Array1;

use crate::filters::weights::Weights;
use crate::utils::binary_search;

// Given a list of filter start and end times, get the weights array.
pub fn get_weights(
    filter_starts: Vec<usize>,
    filter_ends: Vec<usize>,
    frame_start_times: &Array1<usize>,
    start_index: &Array1<usize>,
    array_len: usize,
    include: bool,
) -> Weights {
    let (start_frames, end_frames) = get_indices(frame_start_times, filter_starts, filter_ends);
    get_good_values(start_frames, end_frames, start_index, array_len, include)
}

/// Assuming the data is sorted, get which frames the filters belong to.
#[inline(always)]
fn get_indices(
    start_times: &Array1<usize>,
    filter_starts: Vec<usize>,
    filter_ends: Vec<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let n_filters = filter_starts.len();
    let n_frames = start_times.len();

    // map each filter to a (start, stop) index pair
    (0..n_filters)
        .map(|j| {
            let start = binary_search(start_times, 0, n_frames, filter_starts[j]);
            let mut end = binary_search(start_times, 0, n_frames, filter_ends[j]);
            // if the end is not the top of the array, we should adjust it
            // to be the upper bound of the frame that it is in
            if end + 1 < n_frames {
                end += 1
            }
            (start, end)
        })
        .collect()
}

/// Get a weights array corresponding to the filtered frames.
///
/// Parameters
/// ----------
/// f_start: Vec<usize>
///     The lower bounding frame number for each filter.
/// f_end: Vec<usize>
///     The upper bounding frame number for each filter.
/// start_index: Vec<usize>
///     A list of the first indices for each frame.
/// array_len: usize
///     The length of the final weights array.
/// include: bool
///     Whether the filters represent ranges to include (true) or exclude (false)
///
/// Returns
/// -------
/// Weights
///     An array of the weights corresponding to the filtered frames.
#[inline(always)]
fn get_good_values(
    f_start: Vec<usize>,
    f_end: Vec<usize>,
    start_index: &Array1<usize>,
    array_len: usize,
    include: bool,
) -> Weights {
    // if `include` is true, we start with an array of zeroes and add
    // ranges of ones. if it is false, we start with an array of ones
    // and add ranges of zeroes.
    let mut result = match include {
        true => Weights::zeros(array_len),
        false => Weights::ones(array_len),
    };

    f_start.iter().zip(f_end.iter()).for_each(|(start, end)| {
        result.set_range(start_index[*start], start_index[*end], include);
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that get_indices gets the correct indices.
    #[test]
    fn test_get_indices() {
        let filter_starts = vec![15, 22, 35];
        let filter_ends = vec![20, 25, 41];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);

        let (frame_starts, frame_ends) = get_indices(&start_times, filter_starts, filter_ends);
        assert_eq!(frame_starts, vec![1, 2, 3]);
        assert_eq!(frame_ends, vec![3, 3, 5])
    }

    /// Test that get_indices gets the correct indices when a filter ends above the range.
    #[test]
    fn test_get_indices_above_range() {
        let filter_starts = vec![15];
        let filter_ends = vec![800];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);

        let (frame_starts, frame_ends) = get_indices(&start_times, filter_starts, filter_ends);
        assert_eq!(frame_starts, vec![1]);
        assert_eq!(frame_ends, vec![6])
    }

    /// Test that get_indices gets the correct indices when a filter starts below the range.
    #[test]
    fn test_get_indices_below_range() {
        let filter_starts = vec![2];
        let filter_ends = vec![50];
        let start_times = Array1::from_vec(vec![10, 20, 30, 40, 50, 60, 70]);

        let (frame_starts, frame_ends) = get_indices(&start_times, filter_starts, filter_ends);
        assert_eq!(frame_starts, vec![0]);
        assert_eq!(frame_ends, vec![5])
    }

    /// Test the mask is created correctly for one filter.
    #[test]
    fn test_good_values_one_filter() {
        let f_start = vec![1];
        let f_end = vec![2];
        let start_index = Array1::from_vec(vec![0, 30, 50, 64]);
        let array_len = 64;

        let weights = get_good_values(f_start, f_end, &start_index, array_len, true);

        // expected is 1s between index 30 and 50
        assert_eq!(weights, Weights::from_raw(vec![1125898833100800]))
    }

    /// Test the mask is created correctly for multiple filters.
    #[test]
    fn test_good_values_two_filters() {
        let f_start = vec![1, 4];
        let f_end = vec![2, 6];
        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_good_values(f_start, f_end, &start_index, array_len, true);

        // expected is 1s between indices 10-20 and 40-64
        assert_eq!(weights, Weights::from_raw(vec![18446742974198971392]))
    }

    /// Test the mask is created correctly for two filters that overlap.
    #[test]
    fn test_good_values_overlap() {
        let f_start = vec![1, 3];
        let f_end = vec![4, 5];

        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_good_values(f_start, f_end, &start_index, array_len, true);

        // expected is 1s between indices 10-50
        assert_eq!(weights, Weights::from_raw(vec![1125899906841600]))
    }

    /// Test the mask is created when the filters aren't in increasing order.
    #[test]
    fn test_good_values_out_of_order() {
        let f_start = vec![4, 1];
        let f_end = vec![6, 2];
        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_good_values(f_start, f_end, &start_index, array_len, true);

        // expected is 1s between indices 10-20 and 40-64
        assert_eq!(weights, Weights::from_raw(vec![18446742974198971392]))
    }

    /// Test that the get_weights wrapper function behaves as expected.
    #[test]
    fn test_get_weights_one_filter() {
        let starts = vec![15];
        let ends = vec![31];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_weights(starts, ends, &start_times, &start_index, array_len, true);

        // should be 1s between 10-40
        assert_eq!(weights, Weights::from_raw(vec![1099511626752]))
    }

    /// Test that the get_weights wrapper function behaves as expected for multiple filters.
    #[test]
    fn test_get_weights_two_filters() {
        let starts = vec![15, 41];
        let ends = vec![21, 55];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_weights(starts, ends, &start_times, &start_index, array_len, true);

        // should be 1s between 10-30 and 40-64
        assert_eq!(weights, Weights::from_raw(vec![18446742975271664640]))
    }

    /// Test that the get_weights wrapper function behaves as expected when the filter is entirely
    /// within one frame.
    #[test]
    fn test_get_weights_one_frame() {
        let starts = vec![15];
        let ends = vec![18];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_weights(starts, ends, &start_times, &start_index, array_len, true);

        // should be 1s between 10-20
        assert_eq!(weights, Weights::from_raw(vec![1047552]))
    }

    /// Test that the get_weights wrapper function behaves as expected when the filter is entirely
    /// within the first frame.
    #[test]
    fn test_get_weights_first_frame() {
        let starts = vec![0];
        let ends = vec![8];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_weights(starts, ends, &start_times, &start_index, array_len, true);

        // should be 1s between 0-10
        assert_eq!(weights, Weights::from_raw(vec![0b1111111111]))
    }

    /// Test that the get_weights wrapper function behaves as expected when the filter is entirely
    /// within the last frame.
    #[test]
    fn test_get_weights_last_frame() {
        let starts = vec![55];
        let ends = vec![60];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let start_index = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 64]);
        let array_len = 64;

        let weights = get_weights(starts, ends, &start_times, &start_index, array_len, true);

        // should be 1s between 50-64
        assert_eq!(weights, Weights::from_raw(vec![18445618173802708992]))
    }
}
