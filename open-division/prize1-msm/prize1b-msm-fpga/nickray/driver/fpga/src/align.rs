use core::ops::{Deref, DerefMut};

use crate::{SendBuffer, SendBuffer64};

#[repr(align(64))]
#[derive(Copy, Clone, Debug)]
struct Aligner;

/// Aligns entries to 64 bytes (512 bit) boundaries.
pub struct Aligned<T> {
    // this 0-sized, 64-byte aligned entry aligns the entire struct
    __: [Aligner; 0],
    value: T,
}

impl<T: Clone> Clone for Aligned<T> {
    fn clone(&self) -> Self {
        Self {
            __: [],
            value: self.value.clone(),
        }
    }
}
impl<T: Copy> Copy for Aligned<T> {}

impl<T> Deref for Aligned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for Aligned<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

pub const fn aligned<T>(value: T) -> Aligned<T> {
    Aligned { __: [], value }
}

impl Default for SendBuffer {
    fn default() -> Self {
        aligned([0u8; 64])
    }
}

impl Default for SendBuffer64 {
    fn default() -> Self {
        aligned([0u64; 8])
    }
}
