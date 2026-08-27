use hdf5::types::H5Type;
use ndarray::Array1;

/// Binary search to find the left bounding index of a target value.
/// start and stop are the indices of the array to search between.
#[inline]
pub fn binary_search<T>(array: &Array1<T>, start: usize, stop: usize, target: T) -> usize
where
    T: Ord + Clone,
{
    if stop - start == 1 {
        start
    } else if stop > start {
        let midpoint = start + (stop - start) / 2;
        let midpoint_value = array[midpoint].clone();
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
}
