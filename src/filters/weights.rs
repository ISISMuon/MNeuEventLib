use std::cmp::max;
/// Class representation of an array of binary weights as a string of bits,
/// stored in 64-bit blocks.
///
/// This is used as an efficient way to represent a filter; in the histogram code,
/// a weight of 1 for an event indicates that an event should be included in the histogram,
/// whereas a weight of 0 means it should not be included.
use std::ops::{BitAnd, Index, Not};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

const BLOCK_SIZE: usize = 64;

// the implementation of weights involves some bit manipulation.
// here's a good cheat sheet: https://togglebit.io/posts/rust-bitwise/
// and an explanation for each of the operators: https://www.geeksforgeeks.org/python/python-bitwise-operators/
// (explanation in Python, but it's the same set of operators; only difference is NOT is `!` in Rust
// and `~` in Python)

/// Struct to store weights as a bit string.
/// The bits are stored in 64-bit blocks.
#[derive(Debug, PartialEq)]
pub struct Weights {
    // Note that each block of raw weights is stored in little-endian order;
    // that is, e.g. for the second block (representing weights 64-127)
    // the weight for value 64 would be the rightmost binary digit
    raw_weights: Vec<u64>,
    offset: usize,
}

impl Weights {
    /// Create an array of all ones weights.
    pub fn ones(len: usize) -> Self {
        Weights {
            raw_weights: vec![u64::MAX; max(len / BLOCK_SIZE, 1)],
            offset: 0,
        }
    }

    /// Create an array of all zero weights.
    pub fn zeros(len: usize) -> Self {
        Weights {
            raw_weights: vec![0; max(len / BLOCK_SIZE, 1)],
            offset: 0,
        }
    }

    // this method is used to create expected weights vectors for unit tests,
    // but is not used in the actual code
    #[cfg(test)]
    /// Create a weights array from a raw weight vector.
    #[allow(dead_code)] // dead code here is used in the time filters
    pub fn from_raw(raw_weights: Vec<u64>) -> Self {
        Weights {
            raw_weights,
            offset: 0,
        }
    }

    /// Set the weight at a given index to a given value.
    pub fn set_weight(&mut self, index: usize, set_to: bool) {
        // bit manipulation patterns:
        // `x | 1 << n`     sets bit `n` of the binary number `x` to 1
        // `x & !(1 << n)`  sets bit `n` of the binary number `x` to 0
        // note `1 << n` is the binary number with all zeroes except a 1 at position n,
        // and so `!(1 << n)` is all ones except a zero at position n

        match set_to {
            true => self.raw_weights[index / BLOCK_SIZE] |= 1 << (index % BLOCK_SIZE),
            false => self.raw_weights[index / BLOCK_SIZE] &= !(1 << (index % BLOCK_SIZE)),
        }
    }

    /// Set a range of weights to a given value.
    #[allow(dead_code)] // dead code here is used in the time filters
    pub fn set_range(&mut self, start: usize, end: usize, set_to: bool) {
        // the general idea here is that the range will contain partial and full blocks,
        // so something like
        //
        // |  x--|-----|-----|---x |
        //
        // where | is a block boundary and x-----x is the range.
        // first_block and last_block are the indices of the first
        // and last full blocks contained in the range:
        //
        // |  x--|-----|-----|---x |
        //       ^           ^
        //     first        last
        //
        // so for the partial blocks (start until first_block, and last_block until end)
        // we must individually set each bit. but for full blocks contained in the range,
        // we can just set the entire block with one integer assignment.

        // round start up to the nearest block
        let first_block = match start % BLOCK_SIZE {
            0 => start,
            _ => start + (BLOCK_SIZE - start % BLOCK_SIZE),
        };
        // round end down to the nearest block
        let last_block = match end % BLOCK_SIZE {
            0 => end,
            _ => end - (end % BLOCK_SIZE),
        };

        // if all weights are within one block, just set and exit.
        if start / BLOCK_SIZE == end / BLOCK_SIZE {
            for index in start..end {
                self.set_weight(index, set_to);
            }
            return;
        }

        // set bits individually where we aren't setting the full block
        for index in start..first_block {
            self.set_weight(index, set_to);
        }
        for index in last_block..end {
            self.set_weight(index, set_to);
        }

        // get value to set full blocks to and set full blocks
        let value = match set_to {
            true => u64::MAX,
            false => 0,
        };
        for block in first_block..last_block {
            self.raw_weights[block / BLOCK_SIZE] = value
        }
    }

    /// Get an interval of weights between indices `start` and `end`.
    pub fn slice(&self, start: usize, end: usize) -> Weights {
        // rather than try to copy parts of the blocks and deal with re-chunking the blocks,
        // we just take the full blocks bounding the range given
        // and use an offset to fix the indexing
        // e.g. if the range given is x----x and | is a block boundary:
        //
        // |     |  x--|-----|---x |
        //
        // we copy the full blocks
        //
        // |     |  x--|-----|---x |
        //       ^                 ^
        //
        // and set the offset to 2 so that the 0 index points to the lower x.
        // note that overflow on the right hand side is possible,
        // but we don't iterate over these slices in the histogram code
        // so doesn't happen (we iterate over the times)

        // round start down to the nearest block
        let lower_block_bound = match start % BLOCK_SIZE {
            0 => start,
            _ => start - (start % BLOCK_SIZE),
        };
        // round end up to the nearest block
        let upper_block_bound = match end % BLOCK_SIZE {
            0 => end,
            _ => end + (BLOCK_SIZE - start % BLOCK_SIZE),
        };

        Weights {
            raw_weights: self.raw_weights
                [(lower_block_bound / BLOCK_SIZE)..(upper_block_bound / BLOCK_SIZE)]
                .to_vec(),
            offset: start - lower_block_bound,
        }
    }
}

// allow conversion of iterators of bools into Weights
impl<T: ExactSizeIterator> From<T> for Weights
where
    T::Item: Into<bool>,
{
    fn from(value: T) -> Self {
        let mut result = Weights::zeros(value.len());
        value
            .into_iter()
            .enumerate()
            .for_each(|(k, v)| result.set_weight(k, v.into()));
        result
    }
}

// allow indexing
impl Index<usize> for Weights {
    type Output = bool;

    fn index(&self, index: usize) -> &bool {
        // bit manipulation patterns:
        // `x >> n & 1`

        match (self.raw_weights[index / BLOCK_SIZE + self.offset] >> (index % BLOCK_SIZE)) & 1 {
            1 => &true,
            _ => &false,
        }
    }
}

impl BitAnd for Weights {
    type Output = Weights;

    fn bitand(self, rhs: Self) -> Self::Output {
        // we shouldn't ever need to combine slices, just full weight sets
        if (self.offset != 0) | (rhs.offset != 0) {
            panic!("Can only combine weights with no offset.")
        };

        // we simply iterate bitwise AND over the blocks
        Weights {
            raw_weights: self
                .raw_weights
                .par_iter()
                .zip(rhs.raw_weights.par_iter())
                .map(|(x, y)| x & y)
                .collect(),
            offset: 0,
        }
    }
}

impl Not for Weights {
    type Output = Weights;

    fn not(self) -> Self::Output {
        Weights {
            raw_weights: self.raw_weights.par_iter().map(|x| !x).collect(),
            offset: self.offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // this web link is good for checking that the 'expected values' look how we expect:
    //  https://www.rapidtables.com/convert/number/decimal-to-binary.html
    // we want to look at the signed 2's compliment

    /// Test set_weight sets the expected weights.
    #[test]
    fn test_set_weight() {
        let mut weights = Weights::zeros(128);
        weights.set_weight(15, true);
        weights.set_weight(100, true);
        weights.set_weight(120, true);

        assert_eq!(weights.raw_weights, vec![(1 << 15), (1 << 36) | (1 << 56)]);
    }

    /// Test set_weight works when setting a weight to zero.
    #[test]
    fn test_set_weight_unset() {
        let mut weights = Weights::ones(64);
        weights.set_weight(32, false);
        assert_eq!(weights.raw_weights[0], !(1 << 32));
    }

    /// Test set_range works within one block.
    #[test]
    fn test_set_range_one_block() {
        let mut weights = Weights::zeros(128);
        weights.set_range(30, 50, true);

        // should be values 30 to 50 in block one, then none of block two
        assert_eq!(weights.raw_weights, vec![1125898833100800, 0]);
    }

    /// Test set_range works across blocks.
    #[test]
    fn test_set_range_across_blocks() {
        let mut weights = Weights::zeros(192);
        weights.set_range(30, 150, true);

        // should be values 30 to 64 in block one, all of block two, then 0 to 22 in block three
        assert_eq!(
            weights.raw_weights,
            vec![18446744072635809792, u64::MAX, 4194303]
        );
    }

    /// Test indexing gives the expected bools.
    #[test]
    fn test_indexing() {
        let weights = Weights::from_raw(vec![(1 << 10) | (1 << 32)]);

        (0..64).for_each(|i| match i {
            i if (i == 10) | (i == 32) => assert!(weights[i]),
            _ => assert!(!weights[i]),
        })
    }

    /// Test indexing correctly gets the expected block.
    #[test]
    fn test_indexing_across_blocks() {
        let weights = Weights::from_raw(vec![(1 << 12) | (1 << 20), (1 << 6)]);

        (0..128).for_each(|i| match i {
            i if (i == 12) | (i == 20) | (i == 70) => assert!(weights[i]),
            _ => assert!(!weights[i]),
        })
    }

    /// Test that we can correctly create a weights array from a vector of booleans.
    #[test]
    fn test_from_bool_array() {
        let bools = vec![true, false, true, false].into_iter();
        let weights: Weights = bools.collect::<Vec<bool>>().into_iter().into();

        assert_eq!(weights.raw_weights, vec![0b101])
    }

    /// Test that bitand works for one block.
    #[test]
    fn test_bitand_simple() {
        let weights1 = Weights::from_raw(vec![0b1110]);
        let weights2 = Weights::from_raw(vec![0b1011]);

        let result = weights1 & weights2;
        assert_eq!(result.raw_weights, vec![0b1010]);
    }

    /// Test that bitand works for multiple blocks.
    #[test]
    fn test_bitand_multiple_blocks() {
        let weights1 = Weights::from_raw(vec![0b1110, 0b1001]);
        let weights2 = Weights::from_raw(vec![0b0111, 0b1100]);

        let result = weights1 & weights2;
        assert_eq!(result.raw_weights, vec![0b0110, 0b1000]);
    }

    /// Test that bitand panics if an offset slice is given.
    #[test]
    #[should_panic(expected = "Can only combine weights with no offset")]
    fn test_bitand_with_offset_panics() {
        let mut weights1 = Weights::from_raw(vec![u64::MAX, u64::MAX]);
        let weights2 = Weights::from_raw(vec![u64::MAX, u64::MAX]);

        weights1.offset = 5;
        let _ = weights1 & weights2;
    }

    /// Test the not operator
    #[test]
    fn test_not() {
        let weights = Weights::from_raw(vec![0b1111_0000, 0b1010]);
        let result = !weights;

        assert_eq!(result.raw_weights, vec![!0b1111_0000, !0b1010]);
    }

    /// Test the not operator carries over the offset of the original slice.
    #[test]
    fn test_not_preserves_offset() {
        let mut weights = Weights::from_raw(vec![u64::MAX, u64::MAX]);
        weights.offset = 10;
        let result = !weights;

        assert_eq!(result.offset, 10);
    }

    /// Test slice copies correctly for one block.
    #[test]
    fn test_slice_same_block() {
        let weights = Weights::from_raw(vec![u64::MAX]);
        let sliced = weights.slice(10, 30);

        // Slice should contain the full block but with offset
        assert_eq!(sliced.raw_weights, vec![u64::MAX]);
        assert_eq!(sliced.offset, 10);
    }

    /// Test slice copies correctly for multiple blocks.
    #[test]
    fn test_slice_across_blocks() {
        let weights = Weights::from_raw(vec![u64::MAX, u64::MAX, u64::MAX]);
        let sliced = weights.slice(70, 150);

        assert_eq!(sliced.raw_weights.len(), 2);
        assert_eq!(sliced.offset, 6);
    }

    /// Test slice copies correctly when a full block is given (so no offset)
    #[test]
    fn test_slice_at_boundaries() {
        let weights = Weights::from_raw(vec![u64::MAX, u64::MAX]);
        let sliced = weights.slice(0, 64);

        assert_eq!(sliced.raw_weights.len(), 1);
        assert_eq!(sliced.offset, 0);
    }
}
