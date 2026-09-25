use std::collections::HashMap;

use ndarray::Array1;

use crate::filters::Weights;
use crate::utils::binary_search;

// Given a list of filter start and end times, get the weights array.
#[inline(always)]
pub fn get_weights(
    filter_starts: Vec<usize>,
    filter_ends: Vec<usize>,
    frame_start_times: &Array1<usize>,
    include: bool,
) -> Weights {
    let n_frames = frame_start_times.len();
    let (start_frames, end_frames) = get_indices(frame_start_times, filter_starts, filter_ends);
    get_good_values(start_frames, end_frames, n_frames, include)
}

/// Given a set of log filters and sample logs, get the weights for the log filters.
#[inline(always)]
pub fn get_log_weights(
    filter_times: HashMap<String, (Vec<usize>, Vec<usize>)>,
    frame_start_times: &Array1<usize>,
) -> Weights {
    let n_frames = frame_start_times.len();
    let mut weights = Weights::ones(n_frames);
    for (starts, ends) in filter_times.into_values() {
        let log_weights = get_weights(starts, ends, frame_start_times, true);
        weights = weights & log_weights
    }
    weights
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
            let end = binary_search(start_times, 0, n_frames, filter_ends[j]) + 1;

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
    n_frames: usize,
    include: bool,
) -> Weights {
    // if `include` is true, we start with an array of zeroes and add
    // ranges of ones. if it is false, we start with an array of ones
    // and add ranges of zeroes.
    let mut result = match include {
        true => Weights::zeros(n_frames),
        false => Weights::ones(n_frames),
    };

    f_start.iter().zip(f_end.iter()).for_each(|(start, end)| {
        if end == &n_frames {
            result.set_range(*start, n_frames, include);
        } else {
            result.set_range(*start, *end, include);
        }
    });

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    // NB: recall with the binary numbers in these tests that they are 'indexed' right-to-left
    // (little-endian)

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
        assert_eq!(frame_ends, vec![7])
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
        let f_end = vec![3];

        let weights = get_good_values(f_start, f_end, 4, true);

        assert_eq!(weights, Weights::from_raw(vec![0b0110]))
    }

    /// Test the mask is created correctly for multiple filters.
    #[test]
    fn test_good_values_two_filters() {
        let f_start = vec![1, 4];
        let f_end = vec![2, 7];

        let weights = get_good_values(f_start, f_end, 7, true);

        assert_eq!(weights, Weights::from_raw(vec![0b1110010]))
    }

    /// Test the mask is created correctly for two filters that overlap.
    #[test]
    fn test_good_values_overlap() {
        let f_start = vec![1, 3];
        let f_end = vec![4, 5];

        let weights = get_good_values(f_start, f_end, 7, true);

        assert_eq!(weights, Weights::from_raw(vec![0b0011110]))
    }

    /// Test the mask is created when the filters aren't in increasing order.
    #[test]
    fn test_good_values_out_of_order() {
        let f_start = vec![4, 1];
        let f_end = vec![6, 2];

        let weights = get_good_values(f_start, f_end, 7, true);

        assert_eq!(weights, Weights::from_raw(vec![0b0110010]))
    }

    /// Helper function for get_weights tests.
    fn weight_test_helper(
        starts: Vec<usize>,
        ends: Vec<usize>,
        start_times: Array1<usize>,
        expected: Weights,
    ) {
        let weights = get_weights(starts.clone(), ends.clone(), &start_times, true);
        assert_eq!(weights, expected);

        let weights = get_weights(starts.clone(), ends.clone(), &start_times, false);
        assert_eq!(weights, !expected);
    }

    /// Test that the get_weights wrapper function behaves as expected.
    #[test]
    fn test_get_weights_one_filter() {
        let starts = vec![15];
        let ends = vec![31];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                            ^-------^ filter

        weight_test_helper(
            starts,
            ends,
            start_times,
            Weights::from_raw(vec![0b0001110]),
        )
    }

    /// Test that the get_weights wrapper function behaves as expected for multiple filters.
    #[test]
    fn test_get_weights_two_filters() {
        let starts = vec![15, 41];
        let ends = vec![21, 61];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                            ^--^       ^-------^ filter

        weight_test_helper(
            starts,
            ends,
            start_times,
            Weights::from_raw(vec![0b1110110]),
        )
    }

    /// Test that the get_weights wrapper function behaves as expected when the filter is entirely
    /// within one frame.
    #[test]
    fn test_get_weights_one_frame() {
        let starts = vec![15];
        let ends = vec![18];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                           ^^  filter

        weight_test_helper(
            starts,
            ends,
            start_times,
            Weights::from_raw(vec![0b0000010]),
        )
    }

    /// Test that the get_weights wrapper function behaves as expected when the filter is entirely
    /// within the first frame.
    #[test]
    fn test_get_weights_first_frame() {
        let starts = vec![0];
        let ends = vec![8];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                      ^-^ filter

        weight_test_helper(
            starts,
            ends,
            start_times,
            Weights::from_raw(vec![0b0000001]),
        )
    }

    /// Test that the get_weights wrapper function behaves as expected when the filter is entirely
    /// within the last frame.
    #[test]
    fn test_get_weights_last_frame() {
        let starts = vec![61];
        let ends = vec![63];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                                               ^--^ filter

        weight_test_helper(
            starts,
            ends,
            start_times,
            Weights::from_raw(vec![0b1000000]),
        )
    }

    /// Helper function for get_log_weights tests.
    fn log_filter_times(
        filters: Vec<(&str, Vec<usize>, Vec<usize>)>,
    ) -> HashMap<String, (Vec<usize>, Vec<usize>)> {
        filters
            .into_iter()
            .map(|(name, starts, ends)| (name.to_string(), (starts, ends)))
            .collect()
    }

    /// Test that a single log filter gives the same weights as get_weights.
    #[test]
    fn test_get_log_weights_one_log() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                            ^--------^ log A
        let filter_times = log_filter_times(vec![("A", vec![15], vec![31])]);

        let weights = get_log_weights(filter_times, &start_times);

        assert_eq!(weights, Weights::from_raw(vec![0b0001110]))
    }

    /// Test that the ranges within a single log are combined as a union.
    #[test]
    fn test_get_log_weights_one_log_two_ranges() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                       ^--^       ^--^ log A
        let filter_times = log_filter_times(vec![("A", vec![5, 45], vec![15, 55])]);

        let weights = get_log_weights(filter_times, &start_times);

        assert_eq!(weights, Weights::from_raw(vec![0b0110011]))
    }

    /// Test that two logs are combined as an intersection.
    #[test]
    fn test_get_log_weights_two_logs() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                            ^--------^ log A
        //                                                ^--------^ log B
        let filter_times =
            log_filter_times(vec![("A", vec![15], vec![31]), ("B", vec![25], vec![51])]);

        let weights = get_log_weights(filter_times, &start_times);

        assert_eq!(weights, Weights::from_raw(vec![0b0001100]))
    }

    /// Test that logs with multiple ranges are combined as an intersection of unions.
    #[test]
    fn test_get_log_weights_two_logs_multiple_ranges() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                       ^--^       ^--^ log A
        //                                          ^-----------^ log B
        let filter_times = log_filter_times(vec![
            ("A", vec![5, 45], vec![15, 55]),
            ("B", vec![11], vec![51]),
        ]);

        let weights = get_log_weights(filter_times, &start_times);

        assert_eq!(weights, Weights::from_raw(vec![0b0110010]))
    }

    /// Test that logs whose filters don't overlap give no good frames.
    #[test]
    fn test_get_log_weights_disjoint_logs() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                       ^--^ log A    ^--^ log B
        let filter_times =
            log_filter_times(vec![("A", vec![5], vec![15]), ("B", vec![45], vec![55])]);

        let weights = get_log_weights(filter_times, &start_times);

        assert_eq!(weights, Weights::zeros(7))
    }

    /// Test that a log with no filter ranges gives no good frames.
    #[test]
    fn test_get_log_weights_empty_log() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let filter_times = log_filter_times(vec![("A", vec![], vec![])]);

        let weights = get_log_weights(filter_times, &start_times);

        assert_eq!(weights, Weights::zeros(7))
    }

    /// Test that get_log_weights works when there are more frames than fit in one block.
    #[test]
    fn test_get_log_weights_multiple_blocks() {
        let start_times = Array1::from_vec((0..100).map(|k| k * 10).collect::<Vec<usize>>());
        let filter_times = log_filter_times(vec![
            ("A", vec![105], vec![805]),
            ("B", vec![605], vec![905]),
        ]);

        let weights = get_log_weights(filter_times, &start_times);

        // frames 60 to 80 inclusive are in both logs
        let mut expected = Weights::zeros(100);
        expected.set_range(60, 81, true);
        assert_eq!(weights, expected)
    }
}
