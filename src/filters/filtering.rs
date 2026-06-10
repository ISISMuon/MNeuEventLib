use ndarray::Array1;

use crate::filters::weights::Weights;

/// Binary search to find the left bounding index of a target value.
/// start and stop are the indices of the array to search between.
fn binary_search(
    array: &Array1<usize>,
    start: usize,
    stop: usize,
    target: usize,
) -> Result<usize, ()> {
    if stop - start == 1 {
        Ok(start)
    } else if stop > start {
        let midpoint = start + (stop - start) / 2;
        let midpoint_value = array[midpoint];
        if midpoint_value == target {
            Ok(midpoint)
        } else if midpoint_value > target {
            binary_search(array, start, midpoint, target)
        } else {
            binary_search(array, midpoint, stop, target)
        }
    } else if array[start] < target || array[stop] > target {
        Err(())
    } else {
        Ok(stop)
    }
}

/// Assuming the data is sorted, get which frames the filters belong to.
pub fn get_indices(
    start_times: &Array1<usize>,
    filter_starts: Vec<usize>,
    filter_ends: Vec<usize>,
) -> (Vec<usize>, Vec<usize>) {
    let n_filters = filter_starts.len();
    let n_frames = start_times.len();

    let frame_starts = (0..n_filters)
        .map(|j| {
            binary_search(start_times, 0, n_frames, filter_starts[j])
                .expect("Filter lower bound out of range of data!")
        })
        .collect();
    let frame_ends = (0..n_filters)
        .map(|j| {
            binary_search(start_times, 0, n_frames, filter_ends[j])
                .expect("Filter upper bound out of range of data!")
        })
        .collect();

    (frame_starts, frame_ends)
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
///
pub fn get_good_values(
    f_start: Vec<usize>,
    f_end: Vec<usize>,
    start_index: Array1<usize>,
    array_len: usize,
    include: bool,
) -> Weights {

    let mut result = match include {
        true => Weights::zeros(array_len),
        false => Weights::ones(array_len)
    };

    f_start
        .into_iter()
        .zip(f_end.iter())
        .for_each(|(start, end)| {
            result.set_range(start_index[start], start_index[*end], include);
        });

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the mask is created correctly for one filter.
    #[test]
    fn test_good_values_one_filter() {
        let f_start = vec![1];
        let f_end = vec![2];
        let start_index = Array1::from_vec(vec![0, 30, 50, 64]);
        let array_len = 64;

        let weights = get_good_values(f_start, f_end, start_index, array_len, true);

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

        let weights = get_good_values(f_start, f_end, start_index, array_len, true);

        // expected is 1s between indices 10-20 and 40-64
        assert_eq!(weights, Weights::from_raw(vec![18446742974198971392]))
    }
}
