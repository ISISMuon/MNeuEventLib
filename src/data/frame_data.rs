use ndarray::Array1;

use crate::utils::binary_search;

/// A struct which keeps track of the current frame
/// and the number of values in each frame.
pub struct FrameData {
    pub frame_number: Vec<usize>,
    pub start_index: Array1<usize>,
    pub array_len: usize,
}

impl FrameData {
    /// Create new frame data for a full set of frame start indices.
    pub fn new(start_index: Array1<usize>, array_len: usize) -> FrameData {
        let n_frames = start_index.len();
        FrameData {
            frame_number: (0..n_frames).collect(),
            start_index,
            array_len,
        }
    }

    /// Take a slice of frame data between lower and upper event numbers.
    pub fn slice(&self, lower: usize, upper: usize) -> FrameData {
        let n_frames = self.start_index.len();
        let lower_index = binary_search::<usize>(&self.start_index, 0, n_frames, lower);
        let upper_index = binary_search::<usize>(&self.start_index, 0, n_frames, upper);

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

        FrameData {
            frame_number: (lower_index..=upper_index).collect(),
            start_index: new_starts,
            array_len: upper - lower,
        }
    }
}

/// FrameData routines for tests.
#[cfg(test)]
impl FrameData {
    pub fn one_frame(length: usize) -> FrameData {
        FrameData {
            frame_number: vec![0],
            start_index: Array1::zeros(1),
            array_len: length,
        }
    }

    pub fn one_event_per_frame(length: usize) -> FrameData {
        FrameData {
            frame_number: (0..length).collect(),
            start_index: Array1::from_vec((0..length).collect()),
            array_len: length,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test a new frame data object is successfully created.
    #[test]
    fn test_new() {
        let starts = Array1::from_vec(vec![0, 5, 11, 16]);
        let data = FrameData::new(starts.clone(), 20);

        assert_eq!(data.frame_number, vec![0, 1, 2, 3]);
        assert_eq!(data.start_index, starts);
    }

    /// Test a frame data object is successfully sliced.
    #[test]
    fn test_slice() {
        let starts = Array1::from_vec(vec![0, 5, 11, 16]);
        let data = FrameData::new(starts.clone(), 20);

        let slice = data.slice(7, 15);

        assert_eq!(slice.frame_number, vec![1, 2]);
        assert_eq!(slice.start_index, Array1::from_vec(vec![0, 4]));
        assert_eq!(slice.array_len, 8);
    }
}
