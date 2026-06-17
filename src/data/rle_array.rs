use ndarray::Array1;
/// A run-length encoded array, which stores repeated values more efficiently.
/// e.g. the array [1, 1, 1, 2, 2, 3, 3, 3, 3, 6, 6, 6, 6] would be encoded as
/// values = [1, 2, 3, 6], run_length = [3, 2, 4, 4].
/// Note we don't implement a way to perform this compression; this class exists
/// as a nicer way to deal with the per-frame compression already included in the Nexus data.
pub struct RLEArray {
    pub values: Array1<u32>,
    pub run_lengths: Vec<u32>,
    current_index: usize, // used for iterator impl
    remaining_vals: u32,  // used for iterator impl
}

impl RLEArray {
    /// Construct a new RLEArray from the values for each frame
    /// and the start index of each frame.
    fn from_runs(
        frame_values: Array1<u32>,
        start_index: Array1<u32>,
        array_len: usize,
    ) -> RLEArray {
        let n_frames = frame_values.len();
        let run_lengths: Vec<u32> = (0..n_frames)
            .map(|k| {
                if k == n_frames - 1 {
                    array_len as u32 - start_index[k]
                } else {
                    // get number of values in this frame
                    start_index[k + 1] - start_index[k]
                }
            })
            .collect();
        RLEArray {
            values: frame_values,
            run_lengths,
            current_index: 0,
            remaining_vals: start_index[1],
        }
    }
}

impl Iterator for RLEArray {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index == self.values.len() {
            return None;
        };
        if self.remaining_vals == 0 {
            self.current_index += 1;
            if self.current_index == self.values.len() {
                return None;
            }
            self.remaining_vals = self.run_lengths[self.current_index];
        };
        self.remaining_vals -= 1;
        Some(self.values[self.current_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test a new RLE array is successfully instantiated.
    #[test]
    fn test_new_rle_array() {
        let values = Array1::<u32>::from_vec(vec![1, 2, 3, 4]);
        let start_index = Array1::<u32>::from_vec(vec![0, 4, 9, 16]);

        let arr = RLEArray::from_runs(values.clone(), start_index, 25);
        assert_eq!(arr.values, values);
        assert_eq!(arr.run_lengths, vec![4, 5, 7, 9]);
        assert_eq!(arr.current_index, 0);
        assert_eq!(arr.remaining_vals, 4);
    }

    /// Test iterating over an RLE array correctly decompresses the values.
    #[test]
    fn test_iter() {
        let values = Array1::<u32>::from_vec(vec![1, 2, 3, 4]);
        let start_index = Array1::<u32>::from_vec(vec![0, 2, 5, 7]);

        let arr = RLEArray::from_runs(values.clone(), start_index, 9);

        let full_vec: Vec<u32> = arr.collect();

        assert_eq!(full_vec, vec![1, 1, 2, 2, 2, 3, 3, 4, 4])
    }
}
