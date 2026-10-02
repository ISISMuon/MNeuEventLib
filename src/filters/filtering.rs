use std::cmp::{max, min};

use ndarray::Array1;

use crate::consts::S_TO_NS;
use crate::filters::weights::Weights;
use crate::utils::binary_search;

// Given a list of filter start and end times, get the weights array.
#[inline(always)]
pub fn get_weights(
    filter_starts: Vec<u64>,
    filter_ends: Vec<u64>,
    frame_start_times: &Array1<u64>,
    last_frame_end: u64,
    include: bool,
    label: &str,
) -> Weights {
    let n_frames = frame_start_times.len();
    let n_filters = filter_starts.len();
    let (start_frames, end_frames) = get_indices(
        frame_start_times,
        last_frame_end,
        filter_starts,
        filter_ends,
    );

    // warn if any of the filters are outside of the range
    let n_dropped = n_filters - start_frames.len();
    if n_dropped > 0 {
        let (plural, cover, was) = match n_dropped {
            1 => ("", "covers", "was"),
            _ => ("s", "cover", "were"),
        };
        println!(
            "Warning: {n_dropped} {label}{plural} {cover} no part of the run and {was} ignored. \
             The run spans {:.3}s to {:.3}s.",
            frame_start_times[0] as f64 / S_TO_NS,
            last_frame_end as f64 / S_TO_NS,
        );
    }

    get_good_values(start_frames, end_frames, n_frames, include)
}

/// Assuming the data is sorted, get which frames the filters belong to.
///
/// The run covers the half-open time range [start_times[0], run_end); filters
/// outside it are dropped, and filters partially overlapping it are clamped to it.
#[inline(always)]
fn get_indices(
    start_times: &Array1<u64>,
    run_end: u64,
    filter_starts: Vec<u64>,
    filter_ends: Vec<u64>,
) -> (Vec<usize>, Vec<usize>) {
    let n_filters = filter_starts.len();
    let n_frames = start_times.len();
    let Some(&run_start) = start_times.first() else {
        // no frames, so nothing for any filter to cover
        return (Vec::new(), Vec::new());
    };

    // map each overlapping filter to a (start, stop) index pair
    (0..n_filters)
        .filter_map(|j| {
            // drop filters which are outside the run time
            if filter_ends[j] <= run_start || filter_starts[j] >= run_end {
                return None;
            }

            // trim filters which partially overlap so they are in the run time
            let filter_start = max(filter_starts[j], run_start);
            let filter_end = min(filter_ends[j], run_end - 1);

            let start = binary_search::<u64>(start_times, 0, n_frames, filter_start);
            let end = binary_search::<u64>(start_times, 0, n_frames, filter_end) + 1;

            Some((start, end))
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
        result.set_range(*start, *end, include);
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

        let (frame_starts, frame_ends) = get_indices(&start_times, 70, filter_starts, filter_ends);
        assert_eq!(frame_starts, vec![1, 2, 3]);
        assert_eq!(frame_ends, vec![3, 3, 5])
    }

    /// Test a run with no frames keeps no filters.
    #[test]
    fn test_get_indices_no_frames() {
        let start_times = Array1::from_vec(Vec::<u64>::new());

        let (frame_starts, frame_ends) = get_indices(&start_times, 0, vec![15], vec![20]);
        assert!(frame_starts.is_empty());
        assert!(frame_ends.is_empty())
    }

    /// Test a filter past the end of the run is dropped, rather than snapping to the
    /// last frame (issue #106).
    #[test]
    fn test_get_indices_past_end() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);

        let (frame_starts, frame_ends) = get_indices(&start_times, 70, vec![100], vec![200]);
        assert!(frame_starts.is_empty());
        assert!(frame_ends.is_empty())
    }

    /// Test that get_indices gets the correct indices when a filter ends above the range.
    #[test]
    fn test_get_indices_above_range() {
        let filter_starts = vec![15];
        let filter_ends = vec![800];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);

        let (frame_starts, frame_ends) = get_indices(&start_times, 70, filter_starts, filter_ends);
        assert_eq!(frame_starts, vec![1]);
        assert_eq!(frame_ends, vec![7])
    }

    /// Test that get_indices gets the correct indices when a filter starts below the range.
    #[test]
    fn test_get_indices_below_range() {
        let filter_starts = vec![2];
        let filter_ends = vec![50];
        let start_times = Array1::from_vec(vec![10, 20, 30, 40, 50, 60, 70]);

        let (frame_starts, frame_ends) = get_indices(&start_times, 80, filter_starts, filter_ends);
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
        starts: Vec<u64>,
        ends: Vec<u64>,
        start_times: Array1<u64>,
        run_end: u64,
        expected: Weights,
    ) {
        let weights = get_weights(
            starts.clone(),
            ends.clone(),
            &start_times,
            run_end,
            true,
            "filter",
        );
        assert_eq!(weights, expected);

        let weights = get_weights(
            starts.clone(),
            ends.clone(),
            &start_times,
            run_end,
            false,
            "filter",
        );
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
            70,
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
            70,
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
            70,
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
            70,
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
            70,
            Weights::from_raw(vec![0b1000000]),
        )
    }

    /// Test a filter entirely past the end of the run keeps nothing.
    #[test]
    fn test_get_weights_past_end() {
        let starts = vec![100];
        let ends = vec![200];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                        the run ends at 70, so the filter is well past it

        weight_test_helper(starts, ends, start_times, 70, Weights::from_raw(vec![0]))
    }

    /// Test a filter entirely before the start of the run keeps nothing.
    #[test]
    fn test_get_weights_before_start() {
        let starts = vec![1];
        let ends = vec![5];
        let start_times = Array1::from_vec(vec![10, 20, 30, 40, 50, 60, 70]);
        //                             ^--^ filter, before any frame

        weight_test_helper(starts, ends, start_times, 80, Weights::from_raw(vec![0]))
    }

    /// Test a filter running off the end of the run still keeps the frames it covers.
    #[test]
    fn test_get_weights_overlaps_end() {
        let starts = vec![55];
        let ends = vec![500];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                                        ^------...  filter

        weight_test_helper(
            starts,
            ends,
            start_times,
            70,
            Weights::from_raw(vec![0b1100000]),
        )
    }

    /// Test a filter starting before the run still keeps the frames it covers.
    #[test]
    fn test_get_weights_overlaps_start() {
        let starts = vec![0];
        let ends = vec![15];
        let start_times = Array1::from_vec(vec![10, 20, 30, 40, 50, 60, 70]);
        //                            ...----^ filter

        weight_test_helper(
            starts,
            ends,
            start_times,
            80,
            Weights::from_raw(vec![0b0000001]),
        )
    }

    /// Test dropping an out-of-range filter leaves an in-range one alongside it alone.
    #[test]
    fn test_get_weights_mixed_in_and_out_of_range() {
        let starts = vec![15, 100];
        let ends = vec![31, 200];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                                            ^-------^ filter, then one past the end

        // the same result as test_get_weights_one_filter, which has only the first filter
        weight_test_helper(
            starts,
            ends,
            start_times,
            70,
            Weights::from_raw(vec![0b0001110]),
        )
    }

    /// Test a single-frame run is bounded by the run end it is given.
    #[test]
    fn test_get_weights_one_frame_run() {
        let start_times = Array1::from_vec(vec![100]);
        // the one frame runs from 100 to 200
        let run_end = 200;

        // within the frame
        weight_test_helper(
            vec![150],
            vec![180],
            start_times.clone(),
            run_end,
            Weights::from_raw(vec![0b1]),
        );

        // past it
        weight_test_helper(
            vec![200],
            vec![300],
            start_times.clone(),
            run_end,
            Weights::from_raw(vec![0]),
        );

        // before it
        weight_test_helper(
            vec![0],
            vec![50],
            start_times,
            run_end,
            Weights::from_raw(vec![0]),
        )
    }
}
