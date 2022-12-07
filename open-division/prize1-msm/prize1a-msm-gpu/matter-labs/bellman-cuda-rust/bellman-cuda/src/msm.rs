use ark_bls12_377::{Fq, G1Affine, G1Projective};
use ark_ff::BigInteger256;
use bellman_cuda_cudart::event::CudaEvent;
use bellman_cuda_cudart::execution::{get_raw_fn_and_data, HostFn};
use bellman_cuda_cudart::memory::{CudaMutSlice, CudaSlice, DeviceAllocationMutSlice};
use bellman_cuda_cudart::memory_pools::CudaMemPool;
use bellman_cuda_cudart::result::{CudaResult, CudaResultWrap};
use bellman_cuda_cudart::stream::CudaStream;
use bellman_cuda_sys::*;
use std::fmt::Debug;

#[derive(Copy, Clone, Debug)]
pub struct G1AffineNoInfinity {
    pub x: Fq,
    pub y: Fq,
}

impl From<&G1Affine> for G1AffineNoInfinity {
    fn from(point: &G1Affine) -> Self {
        assert!(!point.infinity);
        G1AffineNoInfinity {
            x: point.x,
            y: point.y,
        }
    }
}

pub fn set_up() -> CudaResult<()> {
    unsafe { msm_set_up().wrap() }
}

pub fn left_shift(
    values: &mut DeviceAllocationMutSlice<G1AffineNoInfinity>,
    shift: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    unsafe {
        msm_left_shift(
            values.as_mut_c_void_ptr(),
            shift,
            values.len() as u32,
            stream.into(),
        )
        .wrap()
    }
}

pub struct ExecuteConfiguration<'a> {
    pub mem_pool: &'a CudaMemPool,
    pub stream: &'a CudaStream,
    pub bases: &'a dyn CudaSlice<G1AffineNoInfinity>,
    pub scalars: &'a dyn CudaSlice<BigInteger256>,
    pub results: &'a mut dyn CudaMutSlice<G1Projective>,
    pub log_scalars_count: u32,
    pub h2d_copy_finished: Option<&'a CudaEvent>,
    pub h2d_copy_finished_callback: Option<&'a HostFn<'a>>,
    pub d2h_copy_finished: Option<&'a CudaEvent>,
    pub d2h_copy_finished_callback: Option<&'a HostFn<'a>>,
    pub force_min_chunk_size: bool,
    pub log_min_chunk_size: u32,
    pub force_max_chunk_size: bool,
    pub log_max_chunk_size: u32,
    pub window_bits_count: u32,
    pub precomputed_windows_stride: u32,
    pub precomputed_bases_stride: u32,
}

pub fn execute_async(configuration: &mut ExecuteConfiguration<'_>) -> CudaResult<()> {
    let h2d_fn_and_data = configuration
        .h2d_copy_finished_callback
        .map(get_raw_fn_and_data);
    let d2h_fn_and_data = configuration
        .d2h_copy_finished_callback
        .map(get_raw_fn_and_data);
    unsafe {
        msm_execute_async(msm_configuration {
            mem_pool: configuration.mem_pool.into(),
            stream: configuration.stream.into(),
            bases: configuration.bases.as_c_void_ptr(),
            scalars: configuration.scalars.as_c_void_ptr(),
            results: configuration.results.as_mut_c_void_ptr(),
            log_scalars_count: configuration.log_scalars_count,
            h2d_copy_finished: configuration
                .h2d_copy_finished
                .map_or(std::mem::zeroed(), |e| e.into()),
            h2d_copy_finished_callback: h2d_fn_and_data.and_then(|f| f.0),
            h2d_copy_finished_callback_data: h2d_fn_and_data.map_or(std::mem::zeroed(), |f| f.1),
            d2h_copy_finished: configuration
                .d2h_copy_finished
                .map_or(std::mem::zeroed(), |e| e.into()),
            d2h_copy_finished_callback: d2h_fn_and_data.and_then(|f| f.0),
            d2h_copy_finished_callback_data: d2h_fn_and_data.map_or(std::mem::zeroed(), |f| f.1),
            force_min_chunk_size: configuration.force_min_chunk_size,
            log_min_chunk_size: configuration.log_min_chunk_size,
            force_max_chunk_size: configuration.force_max_chunk_size,
            log_max_chunk_size: configuration.log_max_chunk_size,
            window_bits_count: configuration.window_bits_count,
            precomputed_windows_stride: configuration.precomputed_windows_stride,
            precomputed_bases_stride: configuration.precomputed_bases_stride,
            scalars_not_montgomery: true,
        })
        .wrap()
    }
}

pub fn tear_down() -> CudaResult<()> {
    unsafe { msm_tear_down().wrap() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_377::Fr;
    use ark_ec::msm::VariableBaseMSM;
    use ark_ec::ProjectiveCurve;
    use ark_ff::{PrimeField, Zero};
    use ark_std::UniformRand;
    use bellman_cuda_cudart::memory::{memory_copy, memory_get_info, DeviceAllocation};
    use bellman_cuda_cudart::memory_pools::{CudaOwnedMemPool, DevicePoolAllocation};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use serial_test::serial;
    use std::ops::{AddAssign, Deref, DerefMut};
    use std::sync::Mutex;

    fn generate_bases(count: usize) -> Vec<G1Projective> {
        let mut rng = ChaCha20Rng::from_entropy();
        (0..count)
            .map(|_| G1Projective::rand(&mut rng))
            .collect::<Vec<_>>()
    }

    fn generate_scalars(count: usize) -> Vec<BigInteger256> {
        let mut rng = ChaCha20Rng::from_entropy();
        (0..count)
            .map(|_| Fr::rand(&mut rng).into_repr())
            .collect::<Vec<_>>()
    }

    fn reduce_results(points: &[G1Projective]) -> G1Projective {
        let mut sum = G1Projective::zero();
        points.iter().rev().for_each(|p| {
            sum.double_in_place().add_assign(p);
        });
        sum
    }

    fn preallocate_pool(pool: &CudaMemPool, log_slack: u32) -> CudaResult<()> {
        let (free, _) = memory_get_info().unwrap();
        let size = ((free >> log_slack) - 1) << log_slack;
        let stream = CudaStream::default();
        let allocation = DevicePoolAllocation::<u8>::alloc_from_pool_async(size, pool, &stream)?;
        allocation.free_async(&stream)?;
        stream.synchronize()?;
        Ok(())
    }

    fn precompute_bases(bases: Vec<G1Projective>, stride: u32, count: u32) -> Vec<G1Projective> {
        let mut result = bases.clone();
        let mut accumulator = bases.clone();
        for _ in 1..count {
            accumulator.iter_mut().for_each(|x| {
                for _ in 0..stride {
                    x.double_in_place();
                }
            });
            result.extend(accumulator.iter());
        }
        result
    }

    #[test]
    #[serial]
    fn msm_simple() {
        set_up().unwrap();
        const LOG_COUNT: u32 = 10;
        const COUNT: usize = 1 << LOG_COUNT;
        const WINDOW_BITS_COUNT: u32 = 23;
        const PRECOMPUTE_FACTOR: u32 = 4;
        const WINDOWS_COUNT: u32 = (253 - 1) / WINDOW_BITS_COUNT + 1;
        const PRECOMPUTE_WINDOWS_STRIDE: u32 = (WINDOWS_COUNT - 1) / PRECOMPUTE_FACTOR + 1;
        const RESULT_BITS_COUNT: u32 = WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;
        let bases = generate_bases(COUNT);
        // let bases = vec![G1Affine::prime_subgroup_generator().into_projective()];
        let bases_2 = precompute_bases(bases, RESULT_BITS_COUNT, PRECOMPUTE_FACTOR);
        // dbg!(&bases_2);
        let bases_2 = G1Projective::batch_normalization_into_affine(&bases_2);
        let bases_no_infinity = bases_2
            .iter()
            .map(G1AffineNoInfinity::from)
            .collect::<Vec<_>>();
        let mut bases_device =
            DeviceAllocation::<G1AffineNoInfinity>::alloc(bases_no_infinity.len()).unwrap();
        memory_copy(&mut bases_device, &bases_no_infinity).unwrap();
        let scalars = generate_scalars(COUNT);
        // let scalars = vec![BigInteger256::from(1)];
        let results = [G1Projective::zero(); RESULT_BITS_COUNT as usize];
        let results_mutex = Mutex::new(results);
        let mem_pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        preallocate_pool(&mem_pool, 25).unwrap();
        let stream = CudaStream::default();
        let reduce_callback = HostFn::new(|| {
            let mut guard = results_mutex.lock().unwrap();
            guard[0] = reduce_results(guard.deref());
        });
        let mut guard = results_mutex.lock().unwrap();
        let mut cfg = ExecuteConfiguration {
            mem_pool: &mem_pool,
            stream: &stream,
            bases: &bases_device,
            scalars: &scalars,
            results: guard.deref_mut(),
            log_scalars_count: LOG_COUNT,
            h2d_copy_finished: None,
            h2d_copy_finished_callback: None,
            d2h_copy_finished: None,
            d2h_copy_finished_callback: Some(&reduce_callback),
            force_min_chunk_size: false,
            log_min_chunk_size: 0,
            force_max_chunk_size: false,
            log_max_chunk_size: 0,
            window_bits_count: WINDOW_BITS_COUNT,
            precomputed_windows_stride: PRECOMPUTE_WINDOWS_STRIDE,
            precomputed_bases_stride: 1u32 << LOG_COUNT,
        };
        execute_async(&mut cfg).unwrap();
        drop(guard);
        stream.synchronize().unwrap();
        let result_gpu = results_mutex.lock().unwrap()[0];
        let result_ark = VariableBaseMSM::multi_scalar_mul(&bases_2, &scalars);
        assert_eq!(result_gpu, result_ark);
    }
}
