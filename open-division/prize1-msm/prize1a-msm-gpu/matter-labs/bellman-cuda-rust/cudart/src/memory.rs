// memory management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

use crate::result::{CudaResult, CudaResultWrap};
use crate::stream::CudaStream;
use bellman_cuda_cudart_sys::*;
use bitflags::bitflags;
use core::ffi::c_void;
use std::alloc::Layout;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice;
use std::slice::SliceIndex;

pub trait CudaSlice<T> {
    /// # Safety
    /// only use for raw pointer operations, do not dereference
    unsafe fn as_slice(&self) -> &[T];

    /// # Safety
    /// only use for raw pointer operations, do not dereference
    unsafe fn as_c_void_ptr(&self) -> *const c_void {
        self.as_slice().as_ptr() as *const c_void
    }
}

pub trait CudaMutSlice<T>: CudaSlice<T> {
    /// # Safety
    /// only use for raw pointer operations, do not dereference
    unsafe fn as_mut_slice(&mut self) -> &mut [T];

    /// # Safety
    /// only use for raw pointer operations, do not dereference
    unsafe fn as_mut_c_void_ptr(&mut self) -> *mut c_void {
        self.as_mut_slice().as_mut_ptr() as *mut c_void
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct DeviceAllocationSlice<'a, T> {
    slice: &'a [T],
}

impl<'a, T> DeviceAllocationSlice<'a, T> {
    fn from_slice(slice: &'a [T]) -> Self {
        Self { slice }
    }

    pub fn index<I>(&self, index: I) -> Self
    where
        I: SliceIndex<[T], Output = [T]>,
    {
        Self::from_slice(self.slice.index(index))
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        self.slice.split_at(mid)
    }
}

impl<T> CudaSlice<T> for DeviceAllocationSlice<'_, T> {
    unsafe fn as_slice(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T> From<&'a DeviceAllocationMutSlice<'a, T>> for DeviceAllocationSlice<'a, T> {
    fn from(slice: &'a DeviceAllocationMutSlice<'a, T>) -> Self {
        Self::from_slice(slice.slice)
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct DeviceAllocationMutSlice<'a, T> {
    pub(crate) slice: &'a mut [T],
}

impl<'a, T> DeviceAllocationMutSlice<'a, T> {
    pub(crate) fn from_slice(slice: &'a mut [T]) -> Self {
        Self { slice }
    }

    pub fn index<'s, I>(&'s self, index: I) -> DeviceAllocationSlice<'s, T>
    where
        I: SliceIndex<[T], Output = [T]>,
    {
        DeviceAllocationSlice::<'s, T>::from_slice(self.slice.index(index))
    }

    pub fn index_mut<'s, I>(&'s mut self, index: I) -> DeviceAllocationMutSlice<'s, T>
    where
        I: SliceIndex<[T], Output = [T]>,
    {
        DeviceAllocationMutSlice::<'s, T>::from_slice(self.slice.index_mut(index))
    }

    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    pub fn split_at<'s>(
        &'s self,
        mid: usize,
    ) -> (DeviceAllocationSlice<'s, T>, DeviceAllocationSlice<'s, T>) {
        let (left, right) = self.slice.split_at(mid);
        (
            DeviceAllocationSlice::from_slice(left),
            DeviceAllocationSlice::from_slice(right),
        )
    }

    pub fn split_at_mut<'s>(
        &'s mut self,
        mid: usize,
    ) -> (
        DeviceAllocationMutSlice<'s, T>,
        DeviceAllocationMutSlice<'s, T>,
    ) {
        let (left, right) = self.slice.split_at_mut(mid);
        (
            DeviceAllocationMutSlice::from_slice(left),
            DeviceAllocationMutSlice::from_slice(right),
        )
    }
}

impl<T> CudaSlice<T> for DeviceAllocationMutSlice<'_, T> {
    unsafe fn as_slice(&self) -> &[T] {
        self.slice
    }
}

impl<T> CudaMutSlice<T> for DeviceAllocationMutSlice<'_, T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct DeviceAllocation<'a, T> {
    slice: DeviceAllocationMutSlice<'a, T>,
}

impl<'a, T> DeviceAllocation<'a, T> {
    fn from_slice(slice: &'a mut [T]) -> Self {
        Self {
            slice: DeviceAllocationMutSlice::<'a, T>::from_slice(slice),
        }
    }

    pub fn alloc(length: usize) -> CudaResult<Self> {
        let layout = Layout::array::<T>(length).unwrap();
        let mut dev_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMalloc(dev_ptr.as_mut_ptr(), layout.size())
                .wrap_maybe_uninit(dev_ptr)
                .map(|ptr| slice::from_raw_parts_mut(ptr as *mut T, length))
                .map(Self::from_slice)
        }
    }

    pub fn free(self) -> CudaResult<()> {
        unsafe {
            let ptr = self.as_c_void_ptr() as *mut c_void;
            mem::forget(self);
            cudaFree(ptr).wrap()
        }
    }
}

impl<T> Drop for DeviceAllocation<'_, T> {
    fn drop(&mut self) {
        let _ = unsafe { cudaFree(self.as_mut_c_void_ptr()) };
    }
}

impl<'a, T> Deref for DeviceAllocation<'a, T> {
    type Target = DeviceAllocationMutSlice<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.slice
    }
}

impl<T> DerefMut for DeviceAllocation<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.slice
    }
}

impl<T> CudaSlice<T> for DeviceAllocation<'_, T> {
    unsafe fn as_slice(&self) -> &[T] {
        self.slice.as_slice()
    }
}

impl<T> CudaMutSlice<T> for DeviceAllocation<'_, T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice.as_mut_slice()
    }
}

bitflags! {
    pub struct CudaHostAllocFlags: u32 {
        const DEFAULT = bellman_cuda_cudart_sys::cudaHostAllocDefault;
        const PORTABLE = bellman_cuda_cudart_sys::cudaHostAllocPortable;
        const MAPPED = bellman_cuda_cudart_sys::cudaHostAllocMapped;
        const WRITE_COMBINED = bellman_cuda_cudart_sys::cudaHostAllocWriteCombined;
    }
}

impl Default for CudaHostAllocFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct HostAllocation<'a, T> {
    slice: &'a mut [T],
}

impl<'a, T> HostAllocation<'a, T> {
    fn from_slice(slice: &'a mut [T]) -> Self {
        Self { slice }
    }

    pub fn alloc(length: usize, flags: CudaHostAllocFlags) -> CudaResult<Self> {
        let layout = Layout::array::<T>(length).unwrap();
        let mut ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaHostAlloc(ptr.as_mut_ptr(), layout.size(), flags.bits)
                .wrap_maybe_uninit(ptr)
                .map(|ptr| slice::from_raw_parts_mut(ptr as *mut T, length))
                .map(Self::from_slice)
        }
    }

    pub fn free(self) -> CudaResult<()> {
        unsafe {
            let ptr = self.as_c_void_ptr() as *mut c_void;
            mem::forget(self);
            cudaFreeHost(ptr).wrap()
        }
    }
}

impl<T> Drop for HostAllocation<'_, T> {
    fn drop(&mut self) {
        let _ = unsafe { cudaFreeHost(self.as_mut_c_void_ptr()) };
    }
}

impl<T> Deref for HostAllocation<'_, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<T> DerefMut for HostAllocation<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

impl<T> AsRef<[T]> for HostAllocation<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.slice
    }
}

impl<T> AsMut<[T]> for HostAllocation<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.slice
    }
}

bitflags! {
    pub struct CudaHostRegisterFlags: u32 {
        const DEFAULT = bellman_cuda_cudart_sys::cudaHostRegisterDefault;
        const PORTABLE = bellman_cuda_cudart_sys::cudaHostRegisterPortable;
        const MAPPED = bellman_cuda_cudart_sys::cudaHostRegisterMapped;
        const IO_MEMORY = bellman_cuda_cudart_sys::cudaHostRegisterIoMemory;
        const READ_ONLY = bellman_cuda_cudart_sys::cudaHostRegisterReadOnly;
    }
}

impl Default for CudaHostRegisterFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct HostRegistration<'a, T> {
    slice: &'a [T],
}

impl<'a, T> HostRegistration<'a, T> {
    fn from_slice(slice: &'a [T]) -> Self {
        Self { slice }
    }

    pub fn register(slice: &'a [T], flags: CudaHostRegisterFlags) -> CudaResult<Self> {
        let length = slice.len();
        let layout = Layout::array::<T>(length).unwrap();
        unsafe {
            let ptr = slice.as_c_void_ptr() as *mut c_void;
            cudaHostRegister(ptr, layout.size(), flags.bits).wrap_value(Self::from_slice(slice))
        }
    }

    pub fn unregister(self) -> CudaResult<()> {
        unsafe { cudaHostUnregister(self.slice.as_c_void_ptr() as *mut c_void).wrap() }
    }
}

impl<T> Drop for HostRegistration<'_, T> {
    fn drop(&mut self) {
        let _ = unsafe { cudaHostUnregister(self.slice.as_c_void_ptr() as *mut c_void).wrap() };
    }
}

impl<T> Deref for HostRegistration<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<T> AsRef<[T]> for HostRegistration<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.slice
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct HostRegistrationMut<'a, T> {
    slice: &'a mut [T],
}

impl<'a, T> HostRegistrationMut<'a, T> {
    fn from_slice(slice: &'a mut [T]) -> Self {
        Self { slice }
    }

    pub fn register(slice: &'a mut [T], flags: CudaHostRegisterFlags) -> CudaResult<Self> {
        let length = slice.len();
        let layout = Layout::array::<T>(length).unwrap();
        unsafe {
            let ptr = slice.as_c_void_ptr() as *mut c_void;
            cudaHostRegister(ptr, layout.size(), flags.bits).wrap_value(Self::from_slice(slice))
        }
    }

    pub fn unregister(self) -> CudaResult<()> {
        unsafe { cudaHostUnregister(self.slice.as_c_void_ptr() as *mut c_void).wrap() }
    }
}

impl<T> Drop for HostRegistrationMut<'_, T> {
    fn drop(&mut self) {
        let _ = unsafe { cudaHostUnregister(self.slice.as_mut_c_void_ptr()).wrap() };
    }
}

impl<T> Deref for HostRegistrationMut<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<T> DerefMut for HostRegistrationMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

impl<T> AsRef<[T]> for HostRegistrationMut<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.slice
    }
}

impl<T> AsMut<[T]> for HostRegistrationMut<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.slice
    }
}

impl<T, U> CudaSlice<U> for T
where
    T: AsRef<[U]>,
{
    unsafe fn as_slice(&self) -> &[U] {
        self.as_ref()
    }
}

impl<T, U> CudaMutSlice<U> for T
where
    T: AsRef<[U]> + AsMut<[U]>,
{
    unsafe fn as_mut_slice(&mut self) -> &mut [U] {
        self.as_mut()
    }
}

pub fn memory_copy<T>(dst: &mut impl CudaMutSlice<T>, src: &impl CudaSlice<T>) -> CudaResult<()> {
    memory_copy_with_kind(dst, src, CudaMemoryCopyKind::Default)
}

pub fn memory_copy_with_kind<T>(
    dst: &mut impl CudaMutSlice<T>,
    src: &impl CudaSlice<T>,
    kind: CudaMemoryCopyKind,
) -> CudaResult<()> {
    unsafe {
        let mut ds = dst.as_mut_slice();
        let ss = src.as_slice();
        assert_eq!(
            ds.len(),
            ss.len(),
            "dst length and src length must be equal"
        );
        let layout = Layout::array::<T>(ds.len()).unwrap();
        cudaMemcpy(
            ds.as_mut_c_void_ptr(),
            ss.as_c_void_ptr(),
            layout.size(),
            kind,
        )
        .wrap()
    }
}

pub fn memory_copy_async<T>(
    dst: &mut impl CudaMutSlice<T>,
    src: &impl CudaSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    memory_copy_with_kind_async(dst, src, CudaMemoryCopyKind::Default, stream)
}

pub fn memory_copy_with_kind_async<T>(
    dst: &mut impl CudaMutSlice<T>,
    src: &impl CudaSlice<T>,
    kind: CudaMemoryCopyKind,
    stream: &CudaStream,
) -> CudaResult<()> {
    unsafe {
        let mut ds = dst.as_mut_slice();
        let ss = src.as_slice();
        assert_eq!(
            ds.len(),
            ss.len(),
            "dst length and src length must be equal"
        );
        let layout = Layout::array::<T>(ds.len()).unwrap();
        cudaMemcpyAsync(
            ds.as_mut_c_void_ptr(),
            ss.as_c_void_ptr(),
            layout.size(),
            kind,
            stream.into(),
        )
        .wrap()
    }
}

pub fn memory_set(dst: &mut impl CudaMutSlice<u8>, value: u8) -> CudaResult<()> {
    unsafe {
        let mut ds = dst.as_mut_slice();
        let layout = Layout::array::<u8>(ds.len()).unwrap();
        cudaMemset(ds.as_mut_c_void_ptr(), value as i32, layout.size()).wrap()
    }
}

pub fn memory_set_async(
    dst: &mut impl CudaMutSlice<u8>,
    value: u8,
    stream: &CudaStream,
) -> CudaResult<()> {
    unsafe {
        let mut ds = dst.as_mut_slice();
        let layout = Layout::array::<u8>(ds.len()).unwrap();
        cudaMemsetAsync(
            ds.as_mut_c_void_ptr(),
            value as i32,
            layout.size(),
            stream.into(),
        )
        .wrap()
    }
}

pub fn memory_get_info() -> CudaResult<(usize, usize)> {
    let mut free = MaybeUninit::<usize>::uninit();
    let mut total = MaybeUninit::<usize>::uninit();
    unsafe {
        let error = cudaMemGetInfo(free.as_mut_ptr(), total.as_mut_ptr());
        if error == CudaError::Success {
            Ok((free.assume_init(), total.assume_init()))
        } else {
            Err(error)
        }
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct HostAllocator {
    flags: CudaHostAllocFlags,
}

impl HostAllocator {
    pub fn new(flags: CudaHostAllocFlags) -> Self {
        Self { flags }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    const LENGTH: usize = 1024;

    #[test]
    #[serial]
    fn device_allocation_alloc_is_ok() {
        let result = DeviceAllocation::<u32>::alloc(LENGTH);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn device_allocation_free_is_ok() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let result = allocation.free();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn device_allocation_alloc_len_eq_length() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        assert_eq!(allocation.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_alloc_is_empty_is_false() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        assert!(!allocation.is_empty());
    }

    #[test]
    #[serial]
    fn device_allocation_deref_len_eq_length() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = allocation.deref();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_deref_mut_len_eq_length() {
        let mut allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = allocation.deref_mut();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_index_len_eq_length() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = allocation.index(..);
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_index_mut_len_eq_length() {
        let mut allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = allocation.index_mut(..);
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_alloc_is_ok() {
        let result = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn host_allocation_free_is_ok() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let result = allocation.free();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn host_allocation_alloc_len_eq_length() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        assert_eq!(allocation.slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_alloc_is_empty_is_false() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        assert!(!allocation.is_empty());
    }

    #[test]
    #[serial]
    fn host_allocation_deref_len_eq_length() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.deref();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_deref_mut_len_eq_length() {
        let mut allocation =
            HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.deref_mut();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_index_len_eq_length() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.index(..);
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_index_mut_len_eq_length() {
        let mut allocation =
            HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.index_mut(..);
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_deref_ptrs_are_equal() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.deref();
        assert_eq!(slice.as_ptr(), allocation.slice.as_ptr());
    }

    #[test]
    #[serial]
    fn host_allocation_deref_mut_ptrs_are_equal() {
        let mut allocation =
            HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.deref_mut();
        assert_eq!(slice.as_mut_ptr(), allocation.slice.as_mut_ptr());
    }

    #[test]
    #[serial]
    fn host_registration_register_is_ok() {
        let mut values = [0u32; LENGTH];
        let result = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        );
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn host_registration_register_empty_error_invalid_value() {
        let mut values = [0u32; 0];
        let result = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        );
        assert_eq!(result.err(), Some(CudaError::ErrorInvalidValue));
    }

    #[test]
    #[serial]
    fn host_registration_unregister_is_ok() {
        let mut values = [0u32; LENGTH];
        let registration = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        let result = registration.unregister();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn host_registration_register_len_eq_length() {
        let mut values = [0u32; LENGTH];
        let registration = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        assert_eq!(registration.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_register_is_empty_is_false() {
        let mut values = [0u32; LENGTH];
        let registration = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        assert!(!registration.is_empty());
    }

    #[test]
    #[serial]
    fn host_registration_deref_len_eq_length() {
        let mut values = [0u32; LENGTH];
        let registration = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        let slice = registration.deref();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_deref_mut_len_eq_length() {
        let mut values = [0u32; LENGTH];
        let mut registration = HostRegistrationMut::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        let slice = registration.deref_mut();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_index_len_eq_length() {
        let mut values = [0u32; LENGTH];
        let registration = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        let slice = registration.index(..);
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_index_mut_len_eq_length() {
        let mut values = [0u32; LENGTH];
        let mut registration = HostRegistrationMut::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = registration.index_mut(..);
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_deref_ptrs_are_equal() {
        let mut values = [0u32; LENGTH];
        let registration = HostRegistration::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        let slice = registration.deref();
        assert_eq!(slice.as_ptr(), registration.slice.as_ptr());
    }

    #[test]
    #[serial]
    fn host_registration_deref_mut_ptrs_are_equal() {
        let mut values = [0u32; LENGTH];
        let mut registration = HostRegistrationMut::<u32>::register(
            values.as_mut_slice(),
            CudaHostRegisterFlags::DEFAULT,
        )
        .unwrap();
        let slice = registration.deref_mut();
        assert_eq!(slice.as_mut_ptr(), registration.slice.as_mut_ptr());
    }

    #[test]
    #[serial]
    fn memory_copy_device_allocation_to_device_allocation() {
        let values1 = [42u32; LENGTH];
        let mut values2 = [0u32; LENGTH];
        let mut a1 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let mut a2 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        memory_copy(&mut a1, &values1).unwrap();
        memory_copy(&mut a2, &a1).unwrap();
        memory_copy(&mut values2, &a2).unwrap();
        assert!(values2.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_host_allocation_to_host_allocation() {
        let mut a1 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut a2 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        a2.iter_mut().for_each(|x| {
            *x = 42u32;
        });
        memory_copy(&mut a1, &a2).unwrap();
        assert!(a1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_host_registration_to_host_registration() {
        let mut values1 = [0u32; LENGTH];
        let mut values2 = [42u32; LENGTH];
        let mut r1 =
            HostRegistrationMut::register(&mut values1, CudaHostRegisterFlags::DEFAULT).unwrap();
        let r2 = HostRegistration::register(&mut values2, CudaHostRegisterFlags::DEFAULT).unwrap();
        memory_copy(&mut r1, &r2).unwrap();
        assert!(r1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_slice_to_slice() {
        let mut values1 = [0u32; LENGTH];
        let values2 = [42u32; LENGTH];
        memory_copy(&mut values1, &values2).unwrap();
        assert!(values1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_device_allocation_to_device_allocation() {
        let stream = CudaStream::create().unwrap();
        let values1 = [42u32; LENGTH];
        let mut values2 = [0u32; LENGTH];
        let mut a1 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let mut a2 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        memory_copy_async(&mut a1, &values1, &stream).unwrap();
        memory_copy_async(&mut a2, &a1, &stream).unwrap();
        memory_copy_async(&mut values2, &a2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(values2.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_host_allocation_to_host_allocation() {
        let stream = CudaStream::create().unwrap();
        let mut a1 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut a2 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        a2.iter_mut().for_each(|x| {
            *x = 42u32;
        });
        memory_copy_async(&mut a1, &a2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(a1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_host_registration_to_host_registration() {
        let stream = CudaStream::create().unwrap();
        let mut values1 = [0u32; LENGTH];
        let mut values2 = [42u32; LENGTH];
        let mut r1 =
            HostRegistrationMut::register(&mut values1, CudaHostRegisterFlags::DEFAULT).unwrap();
        let r2 = HostRegistration::register(&mut values2, CudaHostRegisterFlags::DEFAULT).unwrap();
        memory_copy_async(&mut r1, &r2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(r1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_slice_to_slice() {
        let stream = CudaStream::create().unwrap();
        let mut values1 = [0u32; LENGTH];
        let values2 = [42u32; LENGTH];
        memory_copy_async(&mut values1, &values2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(values1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_set_is_correct() {
        let mut h_values =
            HostAllocation::<u8>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut d_values = DeviceAllocation::<u8>::alloc(LENGTH).unwrap();
        memory_set(&mut d_values, 42u8).unwrap();
        memory_copy(&mut h_values, &d_values).unwrap();
        assert!(h_values.iter().all(|&x| x == 42u8));
    }

    #[test]
    #[serial]
    fn memory_set_async_is_correct() {
        let stream = CudaStream::create().unwrap();
        let mut h_values =
            HostAllocation::<u8>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut d_values = DeviceAllocation::<u8>::alloc(LENGTH).unwrap();
        memory_set_async(&mut d_values, 42u8, &stream).unwrap();
        memory_copy_async(&mut h_values, &d_values, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(h_values.iter().all(|&x| x == 42u8));
    }

    #[test]
    #[serial]
    fn memory_get_info_is_correct() {
        let result = memory_get_info();
        assert!(result.is_ok());
        let (free, total) = result.unwrap();
        assert!(total > 0);
        assert!(free <= total);
    }
}
