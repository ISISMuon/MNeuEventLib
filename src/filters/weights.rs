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
#[derive(Clone, Debug)]
pub struct Weights {
    // Note that each block of raw weights is stored in little-endian order;
    // that is, e.g. for the second block (representing weights 64-127)
    // the weight for value 64 would be the rightmost binary digit
    raw_weights: Vec<u64>,
    // actual length of the array
    len: usize,
}

impl PartialEq for Weights {
    fn eq(&self, other: &Self) -> bool {
        self.raw_weights == other.raw_weights
    }
}

impl Weights {
    /// Create an array of all ones weights.
    pub fn ones(len: usize) -> Self {
        let array_size = match len % BLOCK_SIZE {
            0 => len / BLOCK_SIZE,
            _ => len / BLOCK_SIZE + 1, // account for partial block at end
        };
        Weights {
            raw_weights: vec![u64::MAX; max(array_size, 1)],
            len,
        }
    }

    /// Create an array of all zero weights.
    pub fn zeros(len: usize) -> Self {
        let array_size = match len % BLOCK_SIZE {
            0 => len / BLOCK_SIZE,
            _ => len / BLOCK_SIZE + 1, // account for partial block at end
        };
        Weights {
            raw_weights: vec![0; max(array_size, 1)],
            len,
        }
    }

    /// Set a range of weights to a given value.
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
        // we must create a bit mask, but for full blocks contained in the range,
        // we can just set the entire block with one integer assignment.
        let lower_bit_offset = start % BLOCK_SIZE;
        let upper_bit_offset = end % BLOCK_SIZE;

        // if all weights are within one block, just set and exit.
        if start / BLOCK_SIZE == end / BLOCK_SIZE {
            let block = start / BLOCK_SIZE;

            // mask with 1s in bits [lo, hi)
            // handle hi == BLOCK_SIZE (i.e. 64) specially since `u64::MAX << 64` overflows
            let mask = if upper_bit_offset == BLOCK_SIZE {
                u64::MAX << lower_bit_offset
            } else {
                (u64::MAX << lower_bit_offset) & !(u64::MAX << upper_bit_offset)
            };

            match set_to {
                true => self.raw_weights[block] |= mask,
                false => self.raw_weights[block] &= !mask,
            }
            return;
        }

        // round start up to the nearest block
        let first_block = match lower_bit_offset {
            0 => start,
            _ => start + (BLOCK_SIZE - lower_bit_offset),
        };
        // round end down to the nearest block
        let last_block = match upper_bit_offset {
            0 => end,
            _ => end - (upper_bit_offset),
        };

        // set bits individually where we aren't setting the full block
        if lower_bit_offset != 0 {
            // mask with 1s from lower bit offset to top of block
            let first_block_mask = u64::MAX << lower_bit_offset;
            match set_to {
                true => self.raw_weights[start / BLOCK_SIZE] |= first_block_mask,
                false => self.raw_weights[start / BLOCK_SIZE] &= !first_block_mask,
            }
        }
        if upper_bit_offset != 0 {
            // mask with 1s from upper bit offset to bottom of block
            let last_block_mask = !(u64::MAX << upper_bit_offset);
            match set_to {
                true => self.raw_weights[end / BLOCK_SIZE] |= last_block_mask,
                false => self.raw_weights[end / BLOCK_SIZE] &= !last_block_mask,
            }
        }

        // get value to set full blocks to and set full blocks
        let value = match set_to {
            true => u64::MAX,
            false => 0,
        };
        let lo = first_block / BLOCK_SIZE;
        let hi = last_block / BLOCK_SIZE;
        self.raw_weights[lo..hi].iter_mut().for_each(|b| *b = value);
    }

    /// Count the number of 1s in this array.
    pub fn count(&self) -> u32 {
        let mut count = self
            .raw_weights
            .par_iter()
            .map(|block| (*block).count_ones())
            .sum();
        // subtract any values in the overhang
        let overhang_size = self.len % BLOCK_SIZE;
        if overhang_size != 0 {
            let last_block = self.raw_weights.last().unwrap();
            let overhang = last_block & (u64::MAX << overhang_size);
            count -= overhang.count_ones();
        }
        count
    }

    /// Get the index of the first value set to 1 in the array.
    pub fn get_first_one(&self) -> Option<usize> {
        for (k, block) in self.raw_weights.iter().enumerate() {
            if *block == 0 {
                continue;
            }
            // remember u64 is litte-endian, so the blocks go right-to-left
            return Some(block.trailing_zeros() as usize + (k * BLOCK_SIZE));
        }
        None
    }

    /// Get the index of the last value set to 1 in the array.
    pub fn get_last_one(&self) -> Option<usize> {
        for (k, block) in self.raw_weights.iter().enumerate().rev() {
            if *block == 0 {
                continue;
            }
            // remember u64 is litte-endian, so the blocks go right-to-left
            return Some(block.leading_zeros() as usize + (k * BLOCK_SIZE));
        }
        None
    }
}

// allow indexing
impl Index<usize> for Weights {
    type Output = bool;

    fn index(&self, index: usize) -> &bool {
        // bit manipulation patterns:
        // `x >> n & 1`

        match (self.raw_weights[index / BLOCK_SIZE] >> (index % BLOCK_SIZE)) & 1 {
            1 => &true,
            _ => &false,
        }
    }
}

impl BitAnd for Weights {
    type Output = Weights;

    fn bitand(self, rhs: Self) -> Self::Output {
        // we simply iterate bitwise AND over the blocks
        Weights {
            raw_weights: self
                .raw_weights
                .par_iter()
                .zip(rhs.raw_weights.par_iter())
                .map(|(x, y)| x & y)
                .collect(),
            len: self.len,
        }
    }
}

impl Not for Weights {
    type Output = Weights;

    fn not(self) -> Self::Output {
        Weights {
            raw_weights: self.raw_weights.par_iter().map(|x| !x).collect(),
            len: self.len,
        }
    }
}

/// Routines used for unit tests.
#[cfg(test)]
impl Weights {
    // this method is used to create expected weights vectors for unit tests,
    // but is not used in the actual code
    /// Create a weights array from a raw weight vector.
    pub fn from_raw(raw_weights: Vec<u64>) -> Self {
        Weights {
            raw_weights: raw_weights.clone(),
            len: raw_weights.len() * 64,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    // this web link is good for checking that the 'expected values' look how we expect:
    //  https://www.rapidtables.com/convert/number/decimal-to-binary.html
    // we want to look at the signed 2's compliment

    /// Test that creating an array creates an array of the expected length.
    /// This test is for arrays that are an exact number of blocks.
    #[test]
    fn test_create_array_full_blocks() {
        let ones = Weights::ones(64 * 3);
        let zeros = Weights::zeros(64 * 2);

        assert_eq!(ones.raw_weights, vec![u64::MAX, u64::MAX, u64::MAX]);
        assert_eq!(zeros.raw_weights, vec![0, 0]);
    }

    /// Test that creating an array creates an array of the expected length.
    /// This test is for arrays that are not an exact number of blocks.
    #[test]
    fn test_create_array_partial_blocks() {
        let ones = Weights::ones(200);
        let zeros = Weights::zeros(70);

        assert_eq!(
            ones.raw_weights,
            vec![u64::MAX, u64::MAX, u64::MAX, u64::MAX]
        );
        assert_eq!(zeros.raw_weights, vec![0, 0]);
    }

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

    /// Test the not operator
    #[test]
    fn test_not() {
        let weights = Weights::from_raw(vec![0b1111_0000, 0b1010]);
        let result = !weights;

        assert_eq!(result.raw_weights, vec![!0b1111_0000, !0b1010]);
    }

    /// Test that `count` returns the correct count for one block.
    #[test]
    fn test_count_one_block() {
        let weights = Weights::from_raw(vec![0b01011010001]);
        let count = weights.count();

        assert_eq!(count, 5)
    }

    /// Test that `count` returns the correct count for multiple blocks.
    #[test]
    fn test_count_multiple_blocks() {
        let weights = Weights::from_raw(vec![0b01011010001, 0b111, 0, 0b1]);
        let count = weights.count();

        assert_eq!(count, 9)
    }

    /// Test that `count` returns the correct count when there is an overhang.
    #[test]
    fn test_count_overhang() {
        let weights = Weights::ones(70);
        let count = weights.count();

        assert_eq!(count, 70)
    }

    /// Test that `get_first_one` returns the first 1 value.
    #[test]
    fn test_get_first_one() {
        // remember u64 is litte-endian, so the blocks go right-to-left
        let weights = Weights::from_raw(vec![0, 0b01000100, 0b111000]);

        assert_eq!(weights.get_first_one(), Some(64 + 2))
    }

    #[test]
    fn test_get_last_one() {
        // remember u64 is litte-endian, so the blocks go right-to-left
        let weights = Weights::from_raw(vec![0, 0b01000100, 0b111000]);

        assert_eq!(weights.get_last_one(), Some(128 + (64 - 6)))
    }
}
