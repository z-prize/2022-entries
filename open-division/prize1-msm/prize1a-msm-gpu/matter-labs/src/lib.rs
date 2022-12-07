use ark_bls12_377::{G1Affine, G1Projective};
use ark_ec::ProjectiveCurve;
use ark_ff::BigInteger256;
use ark_std::{log2, Zero};
use bellman_cuda::msm::*;
use bellman_cuda_cudart::event::{CudaEvent, CudaEventCreateFlags};
use bellman_cuda_cudart::execution::*;
use bellman_cuda_cudart::memory::*;
use bellman_cuda_cudart::memory_pools::*;
use bellman_cuda_cudart::stream::{CudaStream, CudaStreamWaitEventFlags};
use bellman_cuda_cudart::unified::pointer_get_attributes;
use bellman_cuda_cudart_sys::CudaMemoryType;
use std::mem::{self, forget};
use std::ops::AddAssign;
use std::sync::{Arc, Mutex};

pub mod util;

const REGISTER_SCALARS: bool = true;
const WINDOW_BITS_COUNT: u32 = 23;
const PRECOMPUTE_FACTOR: u32 = 4;
const WINDOWS_COUNT: u32 = (253 - 1) / WINDOW_BITS_COUNT + 1;
const PRECOMPUTE_WINDOWS_STRIDE: u32 = (WINDOWS_COUNT - 1) / PRECOMPUTE_FACTOR + 1;
const RESULT_BITS_COUNT: u32 = WINDOW_BITS_COUNT * PRECOMPUTE_WINDOWS_STRIDE;

struct Results<'a> {
    unreduced: HostAllocation<'a, G1Projective>,
    reduced: Vec<G1Projective>,
}

impl Results<'_> {
    fn reduce(&mut self) {
        let mut sum = G1Projective::zero();
        self.unreduced
            .iter()
            .rev()
            .for_each(|p| sum.double_in_place().add_assign(p));
        self.reduced.push(sum);
    }
}

struct StreamContext<'a> {
    stream: CudaStream,
    h2d_finished_event: CudaEvent,
    d2h_finished_event: CudaEvent,
    scalars_device: DeviceAllocation<'a, BigInteger256>,
    results: Arc<Mutex<Results<'a>>>,
    reduce_callback: HostFn<'a>,
}

impl StreamContext<'_> {
    fn new(count: usize) -> Self {
        let results = Arc::new(Mutex::new(Results {
            unreduced: HostAllocation::alloc(
                RESULT_BITS_COUNT as usize,
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap(),
            reduced: Vec::new(),
        }));
        let results_clone = results.clone();
        let reduce_callback = HostFn::new(move || results_clone.lock().unwrap().reduce());
        StreamContext {
            stream: CudaStream::create().unwrap(),
            h2d_finished_event: CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)
                .unwrap(),
            d2h_finished_event: CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)
                .unwrap(),
            scalars_device: DeviceAllocation::alloc(count).unwrap(),
            results,
            reduce_callback,
        }
    }
}

pub struct MSMContext<'a> {
    mem_pool: CudaOwnedMemPool,
    log_count: u32,
    bases: DeviceAllocation<'a, G1AffineNoInfinity>,
    stream_context_even: StreamContext<'a>,
    stream_context_odd: StreamContext<'a>,
}

pub fn multi_scalar_mult_init(bases: &[G1Affine]) -> MSMContext {
    let stream = CudaStream::default();
    let mem_pool = CudaOwnedMemPool::create_for_device(0).unwrap();
    mem_pool
        .set_attribute_value(CudaMemPoolAttributeU64::AttrReleaseThreshold, u64::MAX)
        .unwrap();
    let (free, _) = memory_get_info().unwrap();
    const LOG_SLACK: u32 = 25;
    let size = ((free >> LOG_SLACK) - 1) << LOG_SLACK;
    let dummy =
        DevicePoolAllocation::<u8>::alloc_from_pool_async(size, &mem_pool, &stream).unwrap();
    dummy.free_async(&stream).unwrap();
    stream.synchronize().unwrap();
    let count = bases.len();
    let log_count = log2(count);
    assert_eq!(count, 1 << log_count);
    set_up().unwrap();
    let mut bases_device =
        DeviceAllocation::<G1AffineNoInfinity>::alloc(count * PRECOMPUTE_FACTOR as usize).unwrap();
    let bases_no_infinity = bases
        .iter()
        .map(G1AffineNoInfinity::from)
        .collect::<Vec<_>>();
    memory_copy(&mut bases_device.index_mut(0..count), &bases_no_infinity).unwrap();
    for i in 1..PRECOMPUTE_FACTOR as usize {
        let (src, mut dst) = bases_device.split_at_mut(count * i);
        let src = src.index(count * (i - 1)..count * i);
        let mut dst = dst.index_mut(0..count);
        memory_copy_async(&mut dst, &src, &stream).unwrap();
        left_shift(&mut dst, RESULT_BITS_COUNT, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    MSMContext {
        mem_pool,
        log_count,
        bases: bases_device,
        stream_context_even: StreamContext::new(count),
        stream_context_odd: StreamContext::new(count),
    }
}

pub fn multi_scalar_mult(context: &mut MSMContext, _bases: &[G1Affine], scalars: &[BigInteger256]) -> Vec<G1Projective> {
    let scalars_len = scalars.len();
    let mut scalars_type = pointer_get_attributes(&scalars).unwrap().type_;
    if scalars_type == CudaMemoryType::Unregistered && REGISTER_SCALARS {
        forget(HostRegistration::register(&scalars, CudaHostRegisterFlags::default()).unwrap());
        scalars_type = CudaMemoryType::Host;
    };
    let batch_size = scalars_len >> context.log_count;
    assert_eq!(scalars_len, batch_size << context.log_count);
    let mut stream_context = &mut context.stream_context_even;
    let mut other_stream_context = &mut context.stream_context_odd;
    for batch in 0..batch_size {
        let stream = &mut stream_context.stream;
        stream
            .wait_event(
                &other_stream_context.h2d_finished_event,
                CudaStreamWaitEventFlags::DEFAULT,
            )
            .unwrap();
        if batch != 0 {
            let range = (batch << context.log_count)..((batch + 1) << context.log_count);
            memory_copy_async(&mut stream_context.scalars_device, &&scalars[range], stream)
                .unwrap();
        }
        stream
            .wait_event(
                &other_stream_context.d2h_finished_event,
                CudaStreamWaitEventFlags::DEFAULT,
            )
            .unwrap();
        let mut results_guard = stream_context.results.lock().unwrap();
        let mut config = ExecuteConfiguration {
            mem_pool: &context.mem_pool,
            stream,
            bases: &context.bases,
            scalars: if batch == 0 {
                &scalars
            } else {
                &stream_context.scalars_device
            },
            results: &mut results_guard.unreduced,
            log_scalars_count: context.log_count,
            h2d_copy_finished: Some(&stream_context.h2d_finished_event),
            h2d_copy_finished_callback: None,
            d2h_copy_finished: Some(&stream_context.d2h_finished_event),
            d2h_copy_finished_callback: Some(&stream_context.reduce_callback),
            force_min_chunk_size: true,
            log_min_chunk_size: if batch == 0 {
                if scalars_type == CudaMemoryType::Unregistered {
                    22
                } else {
                    23
                }
            } else {
                25
            },
            force_max_chunk_size: true,
            log_max_chunk_size: 25,
            window_bits_count: WINDOW_BITS_COUNT,
            precomputed_windows_stride: PRECOMPUTE_WINDOWS_STRIDE,
            precomputed_bases_stride: 1 << context.log_count,
        };
        execute_async(&mut config).unwrap();
        drop(results_guard);
        mem::swap(&mut stream_context, &mut other_stream_context);
    }
    let mut result = Vec::new();
    let mut stream_context = &mut context.stream_context_even;
    let mut other_stream_context = &mut context.stream_context_odd;
    stream_context.stream.synchronize().unwrap();
    other_stream_context.stream.synchronize().unwrap();
    for batch in 0..batch_size {
        result.push(stream_context.results.lock().unwrap().reduced[batch >> 1]);
        mem::swap(&mut stream_context, &mut other_stream_context);
    }
    result
}
