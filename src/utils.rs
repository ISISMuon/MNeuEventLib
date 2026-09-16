use anyhow::{Error, Result};
use hdf5::types::H5Type;
use ndarray::Array1;
use pyo3::pyfunction;

use crate::BatchData;

/// Binary search to find the left bounding index of a target value.
/// start and stop are the indices of the array to search between.
#[inline]
pub fn binary_search(array: &Array1<usize>, start: usize, stop: usize, target: usize) -> usize {
    if stop - start == 1 {
        start
    } else if stop > start {
        let midpoint = start + (stop - start) / 2;
        let midpoint_value = array[midpoint];
        if midpoint_value == target {
            midpoint
        } else if midpoint_value > target {
            binary_search(array, start, midpoint, target)
        } else {
            binary_search(array, midpoint, stop, target)
        }
    } else if target < array[start] {
        start
    } else {
        stop
    }
}

/// Get the time ranges covered by all filters, non-overlapping.
///
/// The time filter ranges and the log filter ranges are intersected with each
/// other; a set of ranges which is empty because no filters of that kind have
/// been set places no constraint on the other.
///
/// Not designed or maintained for API use. Function to be used by MNeuEventGUI.
#[pyfunction]
pub fn _get_filter_times(index: usize, data: &BatchData) -> Result<(Vec<usize>, Vec<usize>)> {
    let filters = &data.filters[index];
    let data = &data.dataset;
    let frame_times = data.frame_times.read_1d()?;
    let min_time = frame_times[0];
    let max_time = *frame_times.iter().last().unwrap(); // frame times is always finite

    let (mut time_starts, mut time_ends) = filters.get_time_filter_times();
    (time_starts, time_ends) = remove_overlaps(&time_starts, &time_ends);
    if !filters.is_include() && time_starts.len() > 0 {
        (time_starts, time_ends) = invert_intervals(&time_starts, &time_ends, min_time, max_time)
    }

    // get data for all log filters that have been filtered
    let log_names = filters.get_required_log_names();
    let value_logs = match data.get_sample_logs(log_names) {
        Ok(logs) => logs,
        Err(info) => return Err(Error::msg(format!("Failed to get logs: {info}"))),
    };
    let (mut log_starts, mut log_ends) = filters.get_log_filter_times(value_logs);
    (log_starts, log_ends) = remove_overlaps(&log_starts, &log_ends);

    // a kind of filter with no ranges at all isn't constraining anything,
    // so intersecting with it would wrongly wipe out the other kind's ranges
    if time_starts.is_empty() {
        return Ok((log_starts, log_ends));
    }
    if log_starts.is_empty() {
        return Ok((time_starts, time_ends));
    }

    Ok(intersect_intervals(
        &time_starts,
        &time_ends,
        &log_starts,
        &log_ends,
    ))
}

/// Intersect two sets of sorted, disjoint intervals.
fn intersect_intervals(
    starts_a: &[usize],
    ends_a: &[usize],
    starts_b: &[usize],
    ends_b: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut new_starts = Vec::new();
    let mut new_ends = Vec::new();

    // walk both sets in step, taking the overlap of the two current intervals
    let (mut a, mut b) = (0, 0);
    while a < starts_a.len() && b < starts_b.len() {
        let start = starts_a[a].max(starts_b[b]);
        let end = ends_a[a].min(ends_b[b]);
        if start < end {
            new_starts.push(start);
            new_ends.push(end);
        }

        // advance past whichever interval ends first
        if ends_a[a] < ends_b[b] {
            a += 1;
        } else {
            b += 1;
        }
    }

    (new_starts, new_ends)
}

/// Get a list of intervals and remove overlaps.
fn remove_overlaps(starts: &[usize], ends: &[usize]) -> (Vec<usize>, Vec<usize>) {
    if starts.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Pair up starts and ends, then sort by start.
    let mut intervals: Vec<(&usize, &usize)> =
        starts.iter().clone().zip(ends.iter().clone()).collect();

    intervals.sort_by_key(|&(start, _)| start);

    let mut new_starts = Vec::new();
    let mut new_ends = Vec::new();

    let mut current = intervals[0];

    for &(start, end) in &intervals[1..] {
        if start <= current.1 {
            // Overlapping (or touching) interval:
            current.1 = current.1.max(end);
        } else {
            // No overlap, so save the current interval.
            new_starts.push(*current.0);
            new_ends.push(*current.1);
            current = (start, end);
        }
    }

    new_starts.push(*current.0);
    new_ends.push(*current.1);

    (new_starts, new_ends)
}

/// Invert an array of disjoint intervals.
pub fn invert_intervals(
    starts: &[usize],
    ends: &[usize],
    min: usize,
    max: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut new_ends: Vec<usize> = starts.into();
    new_ends.push(max);

    let mut new_starts: Vec<usize> = vec![min];
    new_starts.extend_from_slice(ends);

    (new_starts, new_ends)
}

/// Trait for a 64-bit type that can be converted to a 32-bit one.
pub trait NarrowTo32 {
    type Output: H5Type;
    fn narrow(self) -> Self::Output;
}

impl NarrowTo32 for f64 {
    type Output = f32;
    fn narrow(self) -> f32 {
        self as f32
    }
}

impl NarrowTo32 for i64 {
    type Output = i32;
    fn narrow(self) -> i32 {
        self as i32
    }
}

impl NarrowTo32 for u64 {
    type Output = u32;
    fn narrow(self) -> u32 {
        self as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_utils::MockData;

    /// Create a mock dataset whose frame times are the given times (in ns).
    fn make_mock(frame_times: Vec<usize>) -> MockData {
        let mock = MockData::new().unwrap();
        mock.add_dataset("event_time_zero", Array1::from_vec(frame_times))
            .unwrap();
        mock
    }

    /// Add a sample log to a mock dataset, with times given in seconds.
    fn add_sample_log(mock: &MockData, name: &str, times: Vec<f64>, values: Vec<f64>) {
        let value_log = mock
            .sample_logs
            .create_group(name)
            .unwrap()
            .create_group("value_log")
            .unwrap();
        value_log
            .new_dataset_builder()
            .with_data(&Array1::from_vec(times))
            .create("time")
            .unwrap();
        value_log
            .new_dataset_builder()
            .with_data(&Array1::from_vec(values))
            .create("value")
            .unwrap();
    }

    /// Turn a mock dataset into a BatchData with a single, empty filter set.
    fn make_batch(mock: &MockData) -> BatchData {
        BatchData::from_dataset(mock.create(64, 1048576).unwrap(), 1)
    }

    /// Test the binary search function.
    #[test]
    fn test_binary_search() {
        let array = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let result = binary_search(&array, 0, array.len(), 25);

        assert_eq!(result, 2)
    }

    /// Test the binary search function for a value above the range.
    #[test]
    fn test_binary_search_above_range() {
        let array = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let result = binary_search(&array, 0, array.len(), 80);

        assert_eq!(result, 6)
    }

    /// Test the binary search function for a value below the range.
    #[test]
    fn test_binary_search_below_range() {
        let array = Array1::from_vec(vec![10, 20, 30, 40, 50, 60]);
        let result = binary_search(&array, 0, array.len(), 5);

        assert_eq!(result, 0)
    }

    /// Test that disjoint intervals are left alone by remove_overlaps.
    #[test]
    fn test_remove_overlaps_disjoint() {
        let (starts, ends) = remove_overlaps(&[0, 10, 30], &[5, 15, 35]);

        assert_eq!(starts, vec![0, 10, 30]);
        assert_eq!(ends, vec![5, 15, 35]);
    }

    /// Test that overlapping intervals are merged into one.
    #[test]
    fn test_remove_overlaps_overlapping() {
        let (starts, ends) = remove_overlaps(&[0, 3], &[5, 8]);

        assert_eq!(starts, vec![0]);
        assert_eq!(ends, vec![8]);
    }

    /// Test that intervals which touch at a point are merged into one.
    #[test]
    fn test_remove_overlaps_touching() {
        let (starts, ends) = remove_overlaps(&[0, 5], &[5, 10]);

        assert_eq!(starts, vec![0]);
        assert_eq!(ends, vec![10]);
    }

    /// Test that an interval fully inside another is absorbed by it.
    #[test]
    fn test_remove_overlaps_nested() {
        let (starts, ends) = remove_overlaps(&[0, 2], &[10, 4]);

        assert_eq!(starts, vec![0]);
        assert_eq!(ends, vec![10]);
    }

    /// Test that several overlapping runs of intervals are each merged.
    #[test]
    fn test_remove_overlaps_multiple_merges() {
        let (starts, ends) = remove_overlaps(&[0, 3, 6, 20, 21], &[4, 5, 7, 22, 25]);

        assert_eq!(starts, vec![0, 6, 20]);
        assert_eq!(ends, vec![5, 7, 25]);
    }

    /// Test inverting intervals gives the gaps between them.
    #[test]
    fn test_invert_intervals() {
        let (starts, ends) = invert_intervals(&[2, 10], &[5, 15], 0, 20);

        assert_eq!(starts, vec![0, 5, 15]);
        assert_eq!(ends, vec![2, 10, 20]);
    }

    /// Test inverting no intervals gives the whole range.
    #[test]
    fn test_invert_intervals_empty() {
        let (starts, ends) = invert_intervals(&[], &[], 3, 9);

        assert_eq!(starts, vec![3]);
        assert_eq!(ends, vec![9]);
    }

    /// Test inverting a single interval gives the range either side of it.
    #[test]
    fn test_invert_intervals_single() {
        let (starts, ends) = invert_intervals(&[4], &[6], 0, 10);

        assert_eq!(starts, vec![0, 6]);
        assert_eq!(ends, vec![4, 10]);
    }

    /// Test that a filter set with no filters covers no time at all.
    #[test]
    fn test_get_filter_times_no_filters() {
        let mock = make_mock(vec![0, 5_000_000_000, 10_000_000_000]);
        let batch = make_batch(&mock);

        let (starts, ends) = _get_filter_times(0, &batch).unwrap();

        assert!(starts.is_empty());
        assert!(ends.is_empty());
    }

    /// Test that overlapping time filters are merged.
    #[test]
    fn test_get_filter_times_merges_overlaps() {
        let mock = make_mock(vec![0, 5_000_000_000, 10_000_000_000]);
        let mut batch = make_batch(&mock);
        batch.filters[0]
            .add_time_filter("a".to_string(), 1., 3.)
            .unwrap();
        batch.filters[0]
            .add_time_filter("b".to_string(), 2., 4.)
            .unwrap();

        let (starts, ends) = _get_filter_times(0, &batch).unwrap();

        assert_eq!(starts, vec![1_000_000_000]);
        assert_eq!(ends, vec![4_000_000_000]);
    }

    /// Test that exclude time filters are inverted over the frame time range.
    #[test]
    fn test_get_filter_times_exclude() {
        let mock = make_mock(vec![0, 5_000_000_000, 10_000_000_000]);
        let mut batch = make_batch(&mock);
        batch.filters[0]
            .set_time_type("exclude".to_string())
            .unwrap();
        batch.filters[0]
            .add_time_filter("a".to_string(), 1., 2.)
            .unwrap();

        let (starts, ends) = _get_filter_times(0, &batch).unwrap();

        assert_eq!(starts, vec![0, 2_000_000_000]);
        assert_eq!(ends, vec![1_000_000_000, 10_000_000_000]);
    }

    /// Test that the times a log filter is satisfied are included.
    #[test]
    fn test_get_filter_times_log_filter() {
        let mock = make_mock(vec![0, 5_000_000_000, 10_000_000_000]);
        add_sample_log(
            &mock,
            "temp",
            vec![0., 1., 2., 3., 4.],
            vec![0., 5., 5., 0., 0.],
        );
        let mut batch = make_batch(&mock);
        batch.filters[0]
            .add_log_filter("a".to_string(), "temp".to_string(), Some(4.), Some(6.))
            .unwrap();

        let (starts, ends) = _get_filter_times(0, &batch).unwrap();

        assert_eq!(starts, vec![1_000_000_000]);
        assert_eq!(ends, vec![2_000_000_000]);
    }

    /// Test that only the overlap of the time and log filter times is returned.
    #[test]
    fn test_get_filter_times_time_and_log_filters() {
        let mock = make_mock(vec![0, 5_000_000_000, 10_000_000_000]);
        add_sample_log(
            &mock,
            "temp",
            vec![0., 1., 2., 3., 4.],
            vec![0., 5., 5., 0., 0.],
        );
        let mut batch = make_batch(&mock);
        batch.filters[0]
            .add_time_filter("a".to_string(), 1.5, 7.)
            .unwrap();
        batch.filters[0]
            .add_log_filter("b".to_string(), "temp".to_string(), Some(4.), Some(6.))
            .unwrap();

        let (starts, ends) = _get_filter_times(0, &batch).unwrap();

        assert_eq!(starts, vec![1_500_000_000]);
        assert_eq!(ends, vec![2_000_000_000]);
    }

    /// Test that time and log filters which never overlap cover no time at all.
    #[test]
    fn test_get_filter_times_disjoint_time_and_log_filters() {
        let mock = make_mock(vec![0, 5_000_000_000, 10_000_000_000]);
        add_sample_log(
            &mock,
            "temp",
            vec![0., 1., 2., 3., 4.],
            vec![0., 5., 5., 0., 0.],
        );
        let mut batch = make_batch(&mock);
        batch.filters[0]
            .add_time_filter("a".to_string(), 6., 7.)
            .unwrap();
        batch.filters[0]
            .add_log_filter("b".to_string(), "temp".to_string(), Some(4.), Some(6.))
            .unwrap();

        let (starts, ends) = _get_filter_times(0, &batch).unwrap();

        assert!(starts.is_empty());
        assert!(ends.is_empty());
    }

    /// Test that an exclude time filter is intersected with the log filters.
    #[test]
    fn test_get_filter_times_exclude_and_log_filters() {
        let mock = make_mock(vec![0, 5_000_000_000, 10_000_000_000]);
        add_sample_log(
            &mock,
            "temp",
            vec![0., 1., 2., 3., 4.],
            vec![0., 5., 5., 0., 0.],
        );
        let mut batch = make_batch(&mock);
        batch.filters[0]
            .set_time_type("exclude".to_string())
            .unwrap();
        batch.filters[0]
            .add_time_filter("a".to_string(), 1.5, 7.)
            .unwrap();
        batch.filters[0]
            .add_log_filter("b".to_string(), "temp".to_string(), Some(4.), Some(6.))
            .unwrap();

        let (starts, ends) = _get_filter_times(0, &batch).unwrap();

        assert_eq!(starts, vec![1_000_000_000]);
        assert_eq!(ends, vec![1_500_000_000]);
    }

    /// Test intersecting two sets of intervals which partially overlap.
    #[test]
    fn test_intersect_intervals() {
        let (starts, ends) = intersect_intervals(&[0, 10], &[5, 20], &[3, 12], &[8, 30]);

        assert_eq!(starts, vec![3, 12]);
        assert_eq!(ends, vec![5, 20]);
    }

    /// Test that one interval can intersect several from the other set.
    #[test]
    fn test_intersect_intervals_one_to_many() {
        let (starts, ends) = intersect_intervals(&[0], &[100], &[10, 30, 50], &[20, 40, 60]);

        assert_eq!(starts, vec![10, 30, 50]);
        assert_eq!(ends, vec![20, 40, 60]);
    }

    /// Test that intervals which only touch at a point don't intersect.
    #[test]
    fn test_intersect_intervals_touching() {
        let (starts, ends) = intersect_intervals(&[0], &[5], &[5], &[10]);

        assert!(starts.is_empty());
        assert!(ends.is_empty());
    }

    /// Test that intersecting with no intervals gives no intervals.
    #[test]
    fn test_intersect_intervals_empty() {
        let (starts, ends) = intersect_intervals(&[0, 10], &[5, 20], &[], &[]);

        assert!(starts.is_empty());
        assert!(ends.is_empty());
    }
}
