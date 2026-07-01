use ndarray::Array1;

/// Binary search to find the left bounding index of a target value.
/// start and stop are the indices of the array to search between.
#[inline]
pub fn binary_search(
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
    } else if target < array[start] || array[stop] > target {
        Err(())
    } else {
        Ok(stop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test the binary search function.
    #[test]
    fn test_binary_search() {
        let array = Array1::from_vec(vec![0, 10, 20, 30, 40, 50, 60]);
        let result = binary_search(&array, 0, array.len(), 25);

        assert_eq!(result, Ok(2))
    }
}
