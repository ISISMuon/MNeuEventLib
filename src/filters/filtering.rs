use std::cmp::{max, min};

use ndarray::Array1;

use crate::consts::{FRAME_PERIOD_NS, S_TO_NS};
use crate::filters::weights::Weights;
use crate::utils::binary_search;

// Given a list of filter start and end times, get the weights array.
#[inline(always)]
pub fn get_weights(
    filter_starts: Vec<u64>,
    filter_ends: Vec<u64>,
    frame_start_times: &Array1<u64>,
    include: bool,
    label: &str,
) -> Weights {
    let n_frames = frame_start_times.len();
    let n_filters = filter_starts.len();
    let (start_frames, end_frames) = get_indices(frame_start_times, filter_starts, filter_ends);

    // warn if any of the filters are outside of the range 
    let n_dropped = n_filters - start_frames.len();
    if n_dropped > 0 {
        let (run_start, run_end) = run_range(frame_start_times).unwrap_or((0, 0));
        let (plural, cover, was) = match n_dropped {
            1 => ("", "covers", "was"),
            _ => ("s", "cover", "were"),
        };
        println!(
            "Warning: {n_dropped} {label}{plural} {cover} no part of the run and {was} ignored. \
             The run spans {:.3}s to {:.3}s.",
            run_start as f64 / S_TO_NS,
            run_end as f64 / S_TO_NS,
        );
    }

    get_good_values(start_frames, end_frames, n_frames, include)
}

/// Calculate the half-open time range [start, end) covered by the run's frames,
#[inline(always)]
fn run_range(start_times: &Array1<u64>) -> Option<(u64, u64)> {
    let n_frames = start_times.len();
    let first = *start_times.first()?;
    let last = start_times[n_frames - 1];

    let period = match n_frames {
        // fallback for an empty range 
        1 => FRAME_PERIOD_NS,
        _ => max((last - first) / (n_frames as u64 - 1), 1),
    };

    Some((first, last + period))
}

/// Get the index of the frame containing `target`, or None if `target` falls outside
/// the run.
#[inline(always)]
fn get_frame_index(start_times: &Array1<u64>, run_range: (u64, u64), target: u64) -> Option<usize> {
    let (run_start, run_end) = run_range;
    if target < run_start || target >= run_end {
        return None;
    }
    Some(binary_search::<u64>(
        start_times,
        0,
        start_times.len(),
        target,
    ))
}

/// Assuming the data is sorted, get which frames the filters belong to.
#[inline(always)]
fn get_indices(
    start_times: &Array1<u64>,
    filter_starts: Vec<u64>,
    filter_ends: Vec<u64>,
) -> (Vec<usize>, Vec<usize>) {
    let n_filters = filter_starts.len();
    let Some(range) = run_range(start_times) else {
        // no frames, so nothing for any filter to cover
        return (Vec::new(), Vec::new());
    };
    let (run_start, run_end) = range;

    // map each overlapping filter to a (start, stop) index pair
    (0..n_filters)
        .filter_map(|j| {
            // drop filters which are outside the run time 
            if filter_ends[j] <= run_start || filter_starts[j] >= run_end {
                return None;
            }

            // trim filters which partially overlap so they are in the run time 
            let start = get_frame_index(start_times, range, max(filter_starts[j], run_start))?;
            let end = get_frame_index(start_times, range, min(filter_ends[j], run_end - 1))? + 1;

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

        let (frame_starts, frame_ends) = get_indices(&start_times, filter_starts, filter_ends);
        assert_eq!(frame_starts, vec![1, 2, 3]);
        assert_eq!(frame_ends, vec![3, 3, 5])
    }

    /// Test the run range bounds the last frame with the period measured from the data.
    #[test]
    fn test_run_range() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);

        // the measured period is 10, so the last frame runs from 60 to 70
        assert_eq!(run_range(&start_times), Some((0, 70)))
    }

    /// Test the run range falls back to the nominal period for a single-frame run.
    #[test]
    fn test_run_range_one_frame() {
        let start_times = Array1::from_vec(vec![100]);

        // there is no gap to measure, so the nominal frame period is used
        assert_eq!(run_range(&start_times), Some((100, 100 + FRAME_PERIOD_NS)))
    }

    /// Test a run with no frames has no range at all.
    #[test]
    fn test_run_range_no_frames() {
        let start_times = Array1::from_vec(Vec::<u64>::new());

        assert_eq!(run_range(&start_times), None)
    }

    /// Test a filter past the end of the run is dropped, rather than snapping to the
    /// last frame (issue #106).
    #[test]
    fn test_get_indices_past_end() {
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);

        let (frame_starts, frame_ends) = get_indices(&start_times, vec![100], vec![200]);
        assert!(frame_starts.is_empty());
        assert!(frame_ends.is_empty())
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
        starts: Vec<u64>,
        ends: Vec<u64>,
        start_times: Array1<u64>,
        expected: Weights,
    ) {
        let weights = get_weights(starts.clone(), ends.clone(), &start_times, true, "filter");
        assert_eq!(weights, expected);

        let weights = get_weights(starts.clone(), ends.clone(), &start_times, false, "filter");
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

    /// Test a filter entirely past the end of the run keeps nothing.
    #[test]
    fn test_get_weights_past_end() {
        let starts = vec![100];
        let ends = vec![200];
        let start_times = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        //                        run ends at 70, so the filter is well past it

        weight_test_helper(starts, ends, start_times, Weights::from_raw(vec![0]))
    }

    /// Test a filter entirely before the start of the run keeps nothing.
    #[test]
    fn test_get_weights_before_start() {
        let starts = vec![1];
        let ends = vec![5];
        let start_times = Array1::from_vec(vec![10, 20, 30, 40, 50, 60, 70]);
        //                             ^--^ filter, before any frame

        weight_test_helper(starts, ends, start_times, Weights::from_raw(vec![0]))
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
            Weights::from_raw(vec![0b0001110]),
        )
    }

    /// Test a single-frame run bounds its one frame with the nominal frame period.
    #[test]
    fn test_get_weights_one_frame_run() {
        let start_times = Array1::from_vec(vec![100]);

        // within the nominal period of the one frame
        weight_test_helper(
            vec![100],
            vec![200],
            start_times.clone(),
            Weights::from_raw(vec![0b1]),
        );

        // past it
        weight_test_helper(
            vec![100 + FRAME_PERIOD_NS],
            vec![200 + FRAME_PERIOD_NS],
            start_times.clone(),
            Weights::from_raw(vec![0]),
        );

        // before it
        weight_test_helper(vec![0], vec![50], start_times, Weights::from_raw(vec![0]))
    }
}
