use ndarray::{s, Array1, ArrayView1};

use crate::utils::binary_search;

/// A run-length encoded array, which stores repeated values more efficiently.
/// e.g. the array [1, 1, 1, 2, 2, 3, 3, 3, 3, 6, 6, 6, 6] would be encoded as
/// values = [1, 2, 3, 6], start_index = [0, 3, 5, 9].
/// Note we don't implement a way to perform this compression; this class exists
/// as a nicer way to deal with the per-frame compression already included in the Nexus data.
pub struct RLEArray {
    pub values: Array1<usize>,
    pub start_index: Array1<usize>,
}

/// An iterable slice of an RLEArray.
pub struct RLEArraySlice<'a> {
    pub values: ArrayView1<'a, usize>,
    pub start_index: Array1<usize>,
    pub array_len: usize,  // total size of the uncompressed array
    current_value: usize,  // used for iterator impl: current value being returned
    current_index: usize,  // used for iterator impl: current index of `values`
    remaining_vals: usize, // used for iterator impl: remaining values in this index
}

impl RLEArray {
    /// Construct a new RLEArray from the values for each frame
    /// and the start index of each frame.
    pub fn new(values: Array1<usize>, start_index: Array1<usize>) -> RLEArray {
        RLEArray {
            values,
            start_index,
        }
    }

    /// Create an array of all-zero periods. Used for testing.
    #[cfg(test)]
    pub fn zeros() -> RLEArray {
        RLEArray {
            values: Array1::zeros(1),
            start_index: Array1::zeros(1),
        }
    }

    /// Get a smaller array from a contiguous slice of this one.
    /// `lower` and `upper` correspond to indices on the 'uncompressed' array.
    pub fn slice(&self, lower: usize, upper: usize) -> RLEArraySlice<'_> {
        // find the frames which bound the range
        let n_runs = self.start_index.len();
        let lower_index = binary_search(&self.start_index, 0, n_runs, lower).unwrap();
        let upper_index = binary_search(&self.start_index, 0, n_runs, upper).unwrap();

        // shift the start indices to match the slice
        let new_starts: Array1<usize> = (lower_index..=upper_index)
            .map(|i| {
                if i == lower_index {
                    0
                } else {
                    self.start_index[i] - lower
                }
            })
            .collect();

        let array_len = upper - lower;
        let remaining_vals = match new_starts.len() {
            1 => array_len,
            _ => new_starts[1],
        };

        RLEArraySlice {
            values: self.values.slice(s![lower_index..=upper_index]),
            start_index: new_starts,
            array_len,
            current_value: self.values[lower_index],
            current_index: 0,
            remaining_vals,
        }
    }
}

impl Iterator for RLEArraySlice<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.values.len() {
            return None;
        };
        if self.remaining_vals == 0 {
            self.current_index += 1;
            match self.current_index {
                i if i == self.values.len() => return None,
                i if i == self.values.len() - 1 => {
                    self.remaining_vals = self.array_len - self.start_index[i]
                }
                i => self.remaining_vals = self.start_index[i + 1] - self.start_index[i],
            }
            self.current_value = self.values[self.current_index];
        };
        self.remaining_vals -= 1;
        Some(self.current_value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test a new RLE array is successfully instantiated.
    #[test]
    fn test_new_rle_array() {
        let values = Array1::<usize>::from_vec(vec![1, 2, 3, 4]);
        let start_index = Array1::<usize>::from_vec(vec![0, 4, 9, 16]);

        let arr = RLEArray::new(values.clone(), start_index.clone());
        assert_eq!(arr.values, values);
        assert_eq!(arr.start_index, start_index);
    }

    /// Test iterating over an RLE array correctly decompresses the values.
    #[test]
    fn test_iter() {
        let values = Array1::<usize>::from_vec(vec![1, 2, 3, 4]);
        let start_index = Array1::<usize>::from_vec(vec![0, 2, 8, 9]);

        let arr = RLEArraySlice {
            values: values.view(),
            start_index,
            array_len: 12,
            current_value: 1,
            current_index: 0,
            remaining_vals: 2,
        };

        let full_vec: Vec<usize> = arr.collect();

        assert_eq!(full_vec, vec![1, 1, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4])
    }

    /// Test we can correctly get slices.
    #[test]
    fn test_slice() {
        // decompressed array is
        //
        // [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4]
        //                 [--------------)  slice
        //                 5             10  index
        //
        let values = Array1::<usize>::from_vec(vec![1, 2, 3, 4]);
        let start_index = Array1::<usize>::from_vec(vec![0, 4, 7, 9]);
        let array = RLEArray {
            values,
            start_index,
        };
        let slice = array.slice(5, 10);

        let expected_values = Array1::<usize>::from_vec(vec![2, 3, 4]);
        let expected_idx = Array1::<usize>::from_vec(vec![0, 2, 4]);
        assert_eq!(slice.values, expected_values);
        assert_eq!(slice.start_index, expected_idx);
        assert_eq!(slice.array_len, 5);
    }
}
