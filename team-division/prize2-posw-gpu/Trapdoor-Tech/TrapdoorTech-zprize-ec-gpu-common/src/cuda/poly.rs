/// Implementing a Cuda kernel for polynomial computations
/// Since Cuda context is binded to a CPU thread, our kernel must also save the context while being created
/// NOTE: There are several buffer names reserved by kernel: POWERS_BUFFER, RESULT_BUFFER, BUCKET_BUFFER, FFT_PQ_BUFFER, FFT_OMEGAS_BUFFER
use crate::errors::{GPUError, GPUResult};
use crate::{utils::*, CudaFunction};
use crate::{Context, Device, DeviceMemory, GPUSourceCore, Module, Params, Stream};

use crypto_cuda::make_params;

use log::*;

// ECC types
use crate::structs::PrimeField;

// const values
use crate::log2_floor;
use crate::params::*;

// supported polynomial operations
use crate::PolyArith::*;

use std::collections::HashMap;
use std::sync::Arc;

use std::marker::PhantomData;

/// GpuPoly represents a specific buffer space on GPU that can be viewed as a polynomial, it is used for better abstraction
/// NOTE: Dropping a GpuPoly will trigger cuda memory revocation
/// `F` represents the type that we use for gpu computing
/// `T` represents the type that external code uses
pub struct GpuPoly<'a, 'b, F>
where
    F: PrimeField,
{
    kernel: Arc<&'a PolyKernel<'b, F>>,
    memory: DeviceMemory<F>,
    size: usize,

    fr_name: String,
}

/// `F` for finite field related computation
/// we use `T` for type cast since we may use different cryptography libs
/// type cast works as long as the underlying representations of data are identical
impl<'a, 'b, F> GpuPoly<'a, 'b, F>
where
    F: PrimeField,
{
    /// utilities
    pub fn as_mut_memory(&mut self) -> &mut DeviceMemory<F> {
        &mut self.memory
    }

    /// utilities
    #[inline]
    pub fn get_memory(&self) -> &DeviceMemory<F> {
        &self.memory
    }

    #[inline]
    pub fn get_kernel(&self) -> &'a PolyKernel<'b, F> {
        &self.kernel
    }

    /// utilities
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Managing buffer lifetimes
    pub fn new(kern: &'a PolyKernel<'b, F>, n: usize) -> GPUResult<GpuPoly<'a, 'b, F>> {
        let buffer = kern.create_empty_buffer(n)?;

        let gpu_poly = GpuPoly::<F>::new_from_memory(kern, buffer)?;
        trace!(
            "gpu poly with len {} at {:x} is created",
            n,
            gpu_poly.get_memory().get_inner()
        );

        Ok(gpu_poly)
    }

    /// take a kernel and a me
    pub fn new_from_memory(
        kern: &'a PolyKernel<'b, F>,
        memory: DeviceMemory<F>,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        let n = memory.size();

        let kern_ptr = Arc::new(&*kern);

        let gpu_poly = GpuPoly {
            kernel: kern_ptr,
            memory: memory,
            size: n,

            fr_name: "Fr".to_owned(),
        };

        Ok(gpu_poly)
    }

    /// handy functions for creating specialized buffers
    pub fn new_with_max_len(kern: &'a PolyKernel<'b, F>) -> GPUResult<GpuPoly<'a, 'b, F>> {
        Ok(GpuPoly::<F>::new(&kern, MAX_POLY_LEN)?)
    }

    /// handy functions for creating specialized buffers
    pub fn new_results(kern: &'a PolyKernel<'b, F>) -> GPUResult<GpuPoly<'a, 'b, F>> {
        Ok(GpuPoly::<F>::new(&kern, kern.get_core_x_batch())?)
    }

    /// handy functions for creating specialized buffers
    pub fn new_powers(kern: &'a PolyKernel<'b, F>) -> GPUResult<GpuPoly<'a, 'b, F>> {
        Ok(GpuPoly::<F>::new(&kern, MAX_POLY_LEN_LOG)?)
    }

    /// handy functions for creating specialized buffers
    pub fn new_buckets(kern: &'a PolyKernel<'b, F>) -> GPUResult<GpuPoly<'a, 'b, F>> {
        Ok(GpuPoly::<F>::new(
            &kern,
            kern.get_core_x_batch() * MAX_BUCKET_NUM,
        )?)
    }

    /// handy functions for creating specialized buffers
    pub fn new_pq_buffer(kern: &'a PolyKernel<'b, F>) -> GPUResult<GpuPoly<'a, 'b, F>> {
        Ok(GpuPoly::<F>::new(&kern, 1 << MAX_RADIX_DEGREE >> 1)?)
    }

    /// handy functions for creating specialized buffers
    pub fn new_omegas_buffer(kern: &'a PolyKernel<'b, F>) -> GPUResult<GpuPoly<'a, 'b, F>> {
        Ok(GpuPoly::<F>::new(&kern, LOG2_MAX_ELEMENTS)?)
    }

    pub fn new_omegas_buffer_with_size(kern: &'a PolyKernel<'b, F>, n: usize) -> GPUResult<GpuPoly<'a, 'b, F>> {
        Ok(GpuPoly::<F>::new(&kern, n)?)
    }

    /// Managing buffer lifetimes
    pub fn take_memory(self) -> DeviceMemory<F> {
        self.memory
    }

    /// Memory ops: read from host slice
    /// `T` represents any type which can be converted to `F`
    pub fn read_from<T>(&mut self, poly: &[T]) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = std::cmp::min(poly.len(), self.size);

        let texp = unsafe { std::mem::transmute::<_, &[F]>(poly) };

        self.memory
            .read_from_async(texp, n, self.kernel.get_stream())?;
        //self.kernel.get_stream().sync()?;

        Ok(())
    }

    /// Memory ops: write to host slice
    /// `T` represents any type which can be converted to `F`
    pub fn write_to<T>(&self, poly: &mut [T]) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = std::cmp::min(poly.len(), self.size);

        let texp = unsafe { std::mem::transmute::<_, &mut [F]>(poly) };

        self.memory
            .write_to_async(texp, n, self.kernel.get_stream())?;
        self.kernel.get_stream().sync()?;

        Ok(())
    }

    /// Memory ops: copy from another GpuPoly
    pub fn copy_from_gpu(&mut self, src: &GpuPoly<F>) -> GPUResult<()> {
        if self.memory.get_inner() == src.get_memory().get_inner() {
            return Ok(());
        }

        DeviceMemory::<F>::memcpy_from_to_async(
            src.get_memory(),
            self.get_memory(),
            None,
            self.kernel.get_stream(),
        )?;
        //self.kernel.get_stream().sync()?;

        Ok(())
    }

    /// Memory ops: copy from another GpuPoly
    pub fn copy_from_gpu_with_len(&mut self, src: &GpuPoly<F>, n: usize) -> GPUResult<()> {
        if self.memory.get_inner() == src.get_memory().get_inner() {
            return Ok(());
        }

        DeviceMemory::<F>::memcpy_from_to_async(
            src.get_memory(),
            self.get_memory(),
            Some(n),
            self.kernel.get_stream(),
        )?;
        self.kernel.get_stream().sync()?;

        Ok(())
    }

    /// Memory ops: copy from another GpuPoly
    pub fn copy_from_gpu_offset(&mut self, src: &GpuPoly<F>, offset: usize) -> GPUResult<()> {
        if self.memory.get_inner() == src.get_memory().get_inner() {
            return Ok(());
        }

        let n = self.size;

        assert!(n + offset <= src.size());

        let p1_buf = self.memory.get_inner();
        let p2_buf = src.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, "copy_from_offset_to");
        let params = make_params!(p2_buf, p1_buf, offset as u32, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    // Polynomial Computation
    /// Can only add a short polynomial to a long polynomial, not vice versa
    pub fn add_assign(&mut self, poly: &GpuPoly<F>) -> GPUResult<()> {
        // TODO: is this correct?
        assert!(self.size >= poly.size());

        let n = std::cmp::min(self.size, poly.size());

        let p1_buf = self.get_memory().get_inner();
        let p2_buf = poly.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, AddAssign);
        let params = make_params!(p1_buf, p2_buf, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    /// Can only sub a short polynomial from a long polynomial, not vice versa
    pub fn sub_assign(&mut self, poly: &GpuPoly<F>) -> GPUResult<()> {
        // TODO: is this correct?
        assert!(self.size >= poly.size());

        let n = std::cmp::min(self.size, poly.size());

        let p1_buf = self.get_memory().get_inner();
        let p2_buf = poly.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, SubAssign);
        let params = make_params!(p1_buf, p2_buf, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn mul_assign(&mut self, poly: &GpuPoly<F>) -> GPUResult<()> {
        // assert!(self.size >= poly.size());

        let n = std::cmp::min(self.size, poly.size());

        let p1_buf = self.get_memory().get_inner();
        let p2_buf = poly.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, MulAssign);
        let params = make_params!(p1_buf, p2_buf, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn assign_mul_vanishing(&mut self, poly: &GpuPoly<F>, domain_size: usize) -> GPUResult<()> {
        assert!(self.size >= poly.size() + domain_size);

        let new_size = poly.size() + domain_size;

        let n = poly.size();
        let offset = new_size - n;
        let (gws, lws) = calc_cuda_wg_threads(n);

        let p1_buf = self.memory.get_inner();
        let p2_buf = poly.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, "copy_from_to_offset");
        let params = make_params!(p2_buf, p1_buf, offset as u32, n as u32);
        self.kernel.run(&kernel_name, gws, lws, params)?;

        let kernel_name = format!("{}_poly_{}", self.fr_name, SubAssign);
        let params = make_params!(p1_buf, p2_buf, n as u32);
        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn scale<T>(&mut self, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let n = self.size;

        let p1_buf = self.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, Scaling);
        let params = make_params!(p1_buf, factor, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn add_constant<T>(&mut self, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let n = self.size;

        let p1_buf = self.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, AddConstant);
        let params = make_params!(p1_buf, factor, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn sub_constant<T>(&mut self, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let n = self.size;

        let p1_buf = self.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, SubConstant);
        let params = make_params!(p1_buf, factor, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn add_assign_scale<T>(&mut self, poly: &GpuPoly<F>, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        assert!(self.size >= poly.size());

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let n = std::cmp::min(self.size, poly.size());

        let p1_buf = self.get_memory().get_inner();
        let p2_buf = poly.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, AddAssignScaled);
        let params = make_params!(p1_buf, p2_buf, factor, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn sub_assign_scale<T>(&mut self, poly: &GpuPoly<F>, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        assert!(self.size >= poly.size());

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let n = std::cmp::min(self.size, poly.size());

        let p1_buf = self.get_memory().get_inner();
        let p2_buf = poly.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, SubAssignScaled);
        let params = make_params!(p1_buf, p2_buf, factor, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    /// use `buckets` buffer to save intermediary values
    pub fn batch_inversion(
        &mut self,
        tmp_result: &mut GpuPoly<F>,
        tmp_buckets: &mut GpuPoly<F>,
    ) -> GPUResult<()> {
        let n = self.size;

        let one = F::one();

        let p1_buf = self.memory.get_inner();

        let spawn = std::cmp::min(n, self.kernel.get_core_count() * self.kernel.get_batch());
        let chunk_size = n / spawn + (n % spawn != 0) as usize;

        // keeping inner loop in part_2 small by limiting `bucket_num`, each bucket will have at most `MAX_BUCKET_SIZE` elements
        let bucket_num = std::cmp::min(
            chunk_size / MAX_BUCKET_SIZE + (chunk_size % MAX_BUCKET_SIZE != 0) as usize,
            MAX_BUCKET_NUM,
        );

        assert!(MAX_BUCKET_SIZE * MAX_BUCKET_NUM > chunk_size);

        let mut result = vec![one; spawn];
        let mut subinverses = result.clone();

        let res_buf = tmp_result.get_memory().get_inner();
        let bucket_buf = tmp_buckets.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}_part_1", self.fr_name, BatchInversion);

        let (gws, lws) = calc_cuda_wg_threads(spawn);
        let params = make_params!(
            p1_buf,
            res_buf,
            bucket_buf,
            bucket_num as u32,
            chunk_size as u32,
            n as u32
        );

        self.kernel.run(&kernel_name, gws, lws, params)?;

        tmp_result.write_to(&mut result)?;

        // now coeffs are [a, b, c, d, ..., z]
        // grand_products are [a, ab, abc, d, de, def, ...., xyz]
        // subproducts are [abc, def, xyz]
        // not guaranteed to have equal length
        let mut full_grand_product = one;

        for (acc_r, sub) in result.iter().zip(subinverses.iter_mut()) {
            full_grand_product.mul_assign(acc_r);
            *sub = full_grand_product;
        }

        let product_inverse = full_grand_product.inverse().ok_or(GPUError::Simple(
            "batch inversion division by zero".to_owned(),
        ))?;

        // now let's get [abc^-1, def^-1, ..., xyz^-1];
        let mut acc = product_inverse.clone();

        for i in (1..subinverses.len()).rev() {
            let tmp = subinverses[i - 1].clone();
            subinverses[i] = acc.clone();
            subinverses[i].mul_assign(&tmp);
            acc.mul_assign(&result[i]);
        }

        subinverses[0] = acc;

        tmp_result.read_from(&subinverses[..])?;

        let kernel_name = format!("{}_poly_{}_part_2", self.fr_name, BatchInversion);

        let params = make_params!(
            p1_buf,
            res_buf,
            bucket_buf,
            bucket_num as u32,
            chunk_size as u32,
            n as u32
        );

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    /// use `buckets` buffer to save intermediary values
    pub fn batch_inversion_full(&mut self) -> GPUResult<()> {
        let mut tmp_result = GpuPoly::<F>::new_results(&self.kernel)?;
        let mut tmp_buckets = GpuPoly::<F>::new_buckets(&self.kernel)?;

        self.batch_inversion(&mut tmp_result, &mut tmp_buckets)?;

        Ok(())
    }

    pub fn setup_powers<T>(&mut self, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let pows = (0..MAX_POLY_LEN_LOG)
            .map(|i| factor.pow(&[(1 << i) as u64]))
            .collect::<Vec<_>>();

        self.read_from(&pows)?;

        Ok(())
    }

    /// calculate powers of g by radix 2, then fill a whole buffer with sequential powers of `g`
    pub fn generate_powers<T>(&mut self, powers: &mut GpuPoly<F>, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        powers.setup_powers(c)?;

        let n = self.size;

        let poly_buf = self.memory.get_inner();
        let powers_buf = powers.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, GeneratePowers);
        let params = make_params!(poly_buf, powers_buf, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run_with_sync(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn evaluate_at<T: Copy>(
        &self,
        powers: &mut GpuPoly<F>,
        result: &mut GpuPoly<F>,
        c: &T,
    ) -> GPUResult<T> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = self.size;

        powers.setup_powers(c)?;

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let poly_buf = self.memory.get_inner();
        let powers_buf = powers.get_memory().get_inner();
        let result_buf = result.get_memory().get_inner();

        let spawn = std::cmp::min(n, self.kernel.get_core_count() * self.kernel.get_batch());
        let chunk_size = n / spawn + (n % spawn != 0) as usize;

        let (gws, lws) = calc_cuda_wg_threads(spawn);

        let kernel_name = format!("{}_poly_{}", self.fr_name, EvaluateAt);
        let params = make_params!(
            poly_buf,
            powers_buf,
            result_buf,
            factor,
            chunk_size as u32,
            n as u32
        );

        self.kernel.run(&kernel_name, gws, lws, params)?;

        let mut res = vec![F::one(); spawn];
        result.write_to(&mut res[..])?;

        let mut result = F::zero();
        for v in res.iter() {
            result.add_assign(v);
        }

        let result_converted = unsafe { std::mem::transmute::<_, &T>(&result) };

        Ok(*result_converted)
    }

    /// assuming powers are all setup
    pub fn evaluate_at_naive<T: Copy>(
        &self,
        powers: &mut GpuPoly<F>,
        result: &mut GpuPoly<F>,
        c: &T,
    ) -> GPUResult<T> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = self.size;

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let poly_buf = self.memory.get_inner();
        let powers_buf = powers.get_memory().get_inner();
        let result_buf = result.get_memory().get_inner();

        let spawn = std::cmp::min(n, self.kernel.get_core_count() * self.kernel.get_batch());
        let chunk_size = n / spawn + (n % spawn != 0) as usize;

        let (gws, lws) = calc_cuda_wg_threads(spawn);

        let kernel_name = format!("{}_poly_{}", self.fr_name, EvaluateAt);
        let params = make_params!(
            poly_buf,
            powers_buf,
            result_buf,
            factor,
            chunk_size as u32,
            n as u32
        );

        self.kernel.run(&kernel_name, gws, lws, params)?;

        let mut res = vec![F::one(); spawn];
        result.write_to(&mut res[..])?;

        let mut result = F::zero();
        for v in res.iter() {
            result.add_assign(v);
        }

        let result_converted = unsafe { std::mem::transmute::<_, &T>(&result) };

        Ok(*result_converted)
    }

    pub fn distribute_powers<T>(
        &mut self,
        powers: &mut GpuPoly<F>,
        g: &T,
        offset: usize,
    ) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = self.size;

        powers.setup_powers(g)?;

        let poly_buf = self.memory.get_inner();
        let pow_buf = powers.get_memory().get_inner();

        let &factor = unsafe { std::mem::transmute::<_, &F>(g) };
        let kernel_name = format!("{}_poly_{}", self.fr_name, DistributePowers);

        let spawn = std::cmp::min(n, self.kernel.get_core_count() * self.kernel.get_batch());
        let chunk_size = n / spawn + (n % spawn != 0) as usize;

        let params = make_params!(poly_buf, pow_buf, factor, offset as u32, chunk_size as u32, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(spawn);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn distribute_powers_naive<T>(
        &mut self,
        powers: &GpuPoly<F>,
        g: &T,
        offset: usize,
    ) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = self.size;

        let poly_buf = self.memory.get_inner();
        let pow_buf = powers.get_memory().get_inner();

        let &factor = unsafe { std::mem::transmute::<_, &F>(g) };
        let kernel_name = format!("{}_poly_{}", self.fr_name, DistributePowers);

        let spawn = std::cmp::min(n, self.kernel.get_core_count() * self.kernel.get_batch());
        let chunk_size = n / spawn + (n % spawn != 0) as usize;

        let params = make_params!(poly_buf, pow_buf, factor, offset as u32, chunk_size as u32, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(spawn);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn square(&mut self) -> GPUResult<()> {
        let n = self.size;

        let p1_buf = self.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, Square);
        let params = make_params!(p1_buf, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run_with_sync(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn negate(&mut self) -> GPUResult<()> {
        let n = self.size;

        let p1_buf = self.get_memory().get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, Negate);
        let params = make_params!(p1_buf, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run_with_sync(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    pub fn shift_into(&mut self, dst: &mut GpuPoly<F>, shift: i64) -> GPUResult<()> {
        let n = self.size;

        let src_buf = self.memory.get_inner();
        let dst_buf = dst.get_memory().get_inner();

        let mut shift = shift as i64;

        loop {
            if shift >= n as i64 {
                shift -= n as i64;
            } else if shift < 0 {
                shift += n as i64;
            } else {
                break;
            }
        }

        let kernel_name = format!("{}_poly_{}", self.fr_name, Shift);
        let params = make_params!(src_buf, dst_buf, shift as u32, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run_with_sync(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    /// Shrink a large domain poly evaluations into small domain poly evaluations
    pub fn shrink_domain(&mut self, large: &GpuPoly<F>, factor: usize) -> GPUResult<()> {
        let n = self.size;

        let p1_buf = large.get_memory().get_inner();
        let self_buf = self.memory.get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, ShrinkDomain);
        let params = make_params!(p1_buf, self_buf, factor as u32, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run_with_sync(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    /// Fill a buffer with Field Element
    pub fn fill_with_fe<T>(&mut self, c: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = self.size;

        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let buffer = self.memory.get_inner();

        let kernel_name = format!("{}_poly_{}", self.fr_name, SetFE);
        let params = make_params!(buffer, factor, n as u32);

        let (gws, lws) = calc_cuda_wg_threads(n);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    /// Fill a buffer with Field Element
    pub fn fill_with_zero(&mut self) -> GPUResult<()> {
        let n = self.size;

        self.memory.memset_async(0u32, n, self.kernel.get_stream())?;

        Ok(())
    }

    /// add an Fr at offset
    pub fn add_at_offset<T>(&mut self, c: &T, offset: usize) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let p1_buf = self.memory.get_inner();
        let &factor = unsafe { std::mem::transmute::<_, &F>(c) };

        let kernel_name = format!("{}_poly_{}", self.fr_name, AddAtOffset);

        let params = make_params!(p1_buf, factor, offset as u32);

        let (gws, lws) = calc_cuda_wg_threads(1);

        self.kernel.run(&kernel_name, gws, lws, params)?;

        Ok(())
    }

    /// For FFT, precalculate twiddle factors
    pub fn setup_pq<T>(&mut self, omega: &T, n: usize, max_deg: u32) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        assert!((1 << max_deg >> 1) <= self.size);

        let mut pq = vec![F::zero(); 1 << max_deg >> 1];

        let &omega_f = unsafe { std::mem::transmute::<_, &F>(omega) };

        let tw = omega_f.pow([(n >> max_deg) as u64]);

        pq[0] = F::one();
        if max_deg > 1 {
            pq[1] = tw;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&tw);
            }
        }

        self.read_from(&pq[..])?;

        Ok(())
    }

    /// For FFT, precalculate powers of omegas
    pub fn setup_omegas<T>(&mut self, omega: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        assert!(LOG2_MAX_ELEMENTS <= self.size);

        let &omega_f = unsafe { std::mem::transmute::<_, &F>(omega) };

        let mut omegas = vec![F::zero(); LOG2_MAX_ELEMENTS];

        omegas[0] = omega_f;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }

        self.read_from(&omegas[..])?;

        Ok(())
    }

    pub fn setup_omegas_with_size<T>(&mut self, omega: &T, n: usize) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let &omega_f = unsafe { std::mem::transmute::<_, &F>(omega) };

        let mut omegas = vec![F::zero(); n];

        omegas[0] = F::one();
        for i in 1..n {
            omegas[i] = omegas[i - 1] * omega_f;
        }

        self.read_from(&omegas[..])?;

        Ok(())
    }

    /// Note: after each fft round, the internal DeviceMemory of src/dst buffers are swapped
    fn radix_fft_round(
        &mut self,
        tmp_buffer: &mut GpuPoly<F>,
        pq_buffer: &GpuPoly<F>,
        omegas_buffer: &GpuPoly<F>,
        lgn: u32,
        lgp: u32,
        deg: u32,
        max_deg: u32,
    ) -> GPUResult<()> {
        let src_buf = self.memory.get_inner();
        let dst_buf = tmp_buffer.get_memory().get_inner();

        let fft_pq_buffer = pq_buffer.get_memory().get_inner();
        let fft_omegas_buffer = omegas_buffer.get_memory().get_inner();

        let n = 1u32 << lgn;
        let lwsd = std::cmp::min(max_deg - 1, MAX_LOCAL_WORK_SIZE_DEGREE);

        let lws = 1 << lwsd;
        let gws = (n >> 1) / lws;

        let kernel_name = format!("{}_radix_fft", self.fr_name);
        let params = make_params!(
            src_buf,
            dst_buf,
            fft_pq_buffer,
            fft_omegas_buffer,
            n as u32,
            lgp as u32,
            deg as u32,
            max_deg as u32
        );

        self.kernel.run(&kernel_name, gws, lws, params)?;

        std::mem::swap(&mut self.memory, tmp_buffer.as_mut_memory());

        Ok(())
    }

    /// assuming that pq/omegas are precomputed
    pub fn fft(
        &mut self,
        tmp_buffer: &mut GpuPoly<F>,
        pq_buffer: &GpuPoly<F>,
        omegas_buffer: &GpuPoly<F>,
        lgn: u32,
    ) -> GPUResult<()> {
        let n = self.size();

        assert!(n.is_power_of_two());

        let max_deg = std::cmp::min(MAX_RADIX_DEGREE, lgn);

        let mut lgp = 0u32;

        while lgp < lgn {
            let deg = std::cmp::min(max_deg, lgn - lgp);
            self.radix_fft_round(
                tmp_buffer,
                &pq_buffer,
                &omegas_buffer,
                lgn,
                lgp,
                deg,
                max_deg,
            )?;
            lgp += deg;
        }

        Ok(())
    }

    /// assuming that pq/omegas are precomputed
    pub fn ifft<T>(
        &mut self,
        tmp_buffer: &mut GpuPoly<F>,
        pq_buffer: &GpuPoly<F>,
        omegas_buffer: &GpuPoly<F>,
        size_inv: &T,
        lgn: u32,
    ) -> GPUResult<()> {
        self.fft(tmp_buffer, pq_buffer, omegas_buffer, lgn)?;

        self.scale(size_inv)?;

        Ok(())
    }

    /// do a full FFT on self buffer, including computing all the necessary pq/omegas
    pub fn fft_full<T>(&mut self, omega: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = self.size();
        let lgn = log2_floor(n);

        let mut tmp_buffer = GpuPoly::<F>::new(self.get_kernel().clone(), n)?;

        let mut pq_buffer = GpuPoly::<F>::new_pq_buffer(self.get_kernel().clone())?;

        let mut omegas_buffer = GpuPoly::<F>::new_omegas_buffer_with_size(self.get_kernel().clone(), n)?;

        let max_deg = std::cmp::min(MAX_RADIX_DEGREE, lgn);

        pq_buffer.setup_pq(omega, n, max_deg)?;
        omegas_buffer.setup_omegas_with_size(omega, n)?;

        self.fft(&mut tmp_buffer, &pq_buffer, &omegas_buffer, lgn)?;

        Ok(())
    }

    /// do a full iFFT on self buffer, including computing all the necessary pq/omegas
    pub fn ifft_full<T>(&mut self, omega_inv: &T) -> GPUResult<()> {
        debug_assert!(std::mem::size_of::<F>() == std::mem::size_of::<T>());

        let n = self.size();
        let lgn = log2_floor(n);

        let mut tmp_buffer = GpuPoly::<F>::new(self.get_kernel().clone(), n)?;

        let mut pq_buffer = GpuPoly::<F>::new_pq_buffer(self.get_kernel().clone())?;

        let mut omegas_buffer = GpuPoly::<F>::new_omegas_buffer_with_size(self.get_kernel().clone(), n)?;

        let max_deg = std::cmp::min(MAX_RADIX_DEGREE, lgn);

        pq_buffer.setup_pq(omega_inv, n, max_deg)?;
        omegas_buffer.setup_omegas_with_size(omega_inv, n)?;

        let size_inv = F::from_str(&format!("{}", self.size)).map_err(|_| GPUError::Simple("cannot get minv: incorrect field element".to_owned()))?;
        let size_inv = size_inv.inverse().expect("m must have inverse");

        self.ifft(&mut tmp_buffer, &pq_buffer, &omegas_buffer, &size_inv, lgn)?;

        Ok(())
    }
}

/// PolyKernel represents a initialized cuda context that can manipulate GpuPoly
pub struct PolyKernel<'a, F>
where
    F: PrimeField,
{
    device: Device,
    module: Module,
    context: Arc<&'a Context>,
    stream: Stream,
    kernel_functions: HashMap<String, CudaFunction>,

    batch: usize,
    core_count: usize,

    _phantom: PhantomData<F>,
}

impl<'a, F> PolyKernel<'a, F>
where
    F: PrimeField,
{
    pub fn get_context(&self) -> &Context {
        &self.context
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_module(&self) -> &Module {
        &self.module
    }

    pub fn get_stream(&self) -> &Stream {
        &self.stream
    }

    pub fn get_core_count(&self) -> usize {
        self.core_count
    }

    pub fn get_batch(&self) -> usize {
        self.batch
    }

    pub fn run_with_sync(&self, func: &str, gws: u32, lws: u32, params: &Params) -> GPUResult<()> {
        let f = self.kernel_functions.get(func).unwrap().inner;

        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;
        self.stream.sync()?;

        Ok(())
    }

    pub fn run(&self, func: &str, gws: u32, lws: u32, params: &Params) -> GPUResult<()> {
        let f = self.kernel_functions.get(func).unwrap().inner;

        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        Ok(())
    }

    pub fn sync(&self) -> GPUResult<()> {
        self.stream.sync()?;
        Ok(())
    }

    pub fn create_with_core(core: &'a GPUSourceCore) -> GPUResult<Self> {
        let time = std::time::Instant::now();

        let context = Arc::new(&core.context);
        let device = core.device.clone();
        let module = core.module.clone();
        let kernel_functions = core.kernel_func_map.clone();

        let core_count = device.get_cores()?;
        let batch = CUDA_BEST_BATCH;

        let stream = Stream::new_with_context(*context)?;

        info!("initialize cuda kernel time = {:?}", time.elapsed());

        Ok(Self {
            device,
            module,
            context,
            stream,
            kernel_functions,

            batch,
            core_count,

            _phantom: PhantomData,
        })
    }

    pub fn create_empty_buffer(&self, n: usize) -> GPUResult<DeviceMemory<F>> {
        if n > (1 << LOG2_MAX_ELEMENTS) {
            return Err(GPUError::Simple(format!(
                "required buffer for [{}] elements is too big!",
                n
            )));
        }

        let buffer = DeviceMemory::<F>::new(&*self.context, n)?;

        trace!("buffer containing [{}] elements is created", n);

        Ok(buffer)
    }

    pub fn get_core_x_batch(&self) -> usize {
        self.core_count * self.batch
    }
}
