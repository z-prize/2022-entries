// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

use snarkvm_curves::{
    bls12_377::{Fr, G1Affine, G1Projective},
    traits::AffineCurve,
};
use snarkvm_fields::Zero;

use std::any::TypeId;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use ec_gpu_common::{
    DeviceMemory, G1Affine as GpuG1Affine, G1Projective as GpuG1Projective, GPUError, GpuFr, MsmPrecalcContainer,
    MultiexpKernel, Zero as OtherZero, GPU_OP,
};

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

pub enum GpuScalars<'a> {
    Slice(Arc<&'a [Fr]>),
    Memory(&'a DeviceMemory<GpuFr>),
}

pub struct MSMRequest<'a> {
    bases: Arc<&'a [G1Affine]>,
    scalars: GpuScalars<'a>,
    start: usize,
    n: usize,
    gpu_op: GPU_OP,
    is_mont_form: bool,
    response: crossbeam_channel::Sender<Result<G1Projective, GPUError>>,
}

lazy_static::lazy_static! {
    static ref ACC_POINTS: Arc<RwLock<Vec<Vec<G1Projective>>>> = Arc::new(RwLock::new(vec![Vec::<G1Projective>::new(); 4]));

    static ref WINDOW_SIZE: AtomicUsize = AtomicUsize::new(0);

    static ref MSM_DISPATCH: crossbeam_channel::Sender<MSMRequest<'static>> = {
        let (sender, receiver) = crossbeam_channel::bounded(100);

        let gpu_idx: usize = {
            std::env::var("GPU_IDX")
                .and_then(|v| match v.parse() {
                    Ok(val) => Ok(val),
                    Err(_) => {
                        println!("Invalid env GPU_IDX! Defaulting to 0...");
                        Ok(0)
                    }
                })
                .unwrap_or(0)
        };

        let precalc_container = MsmPrecalcContainer::create(gpu_idx).unwrap();
        let precalc_container_arc = Arc::new(precalc_container);

        let empiric_thread_count = 8;

        for i in 0..empiric_thread_count {
            let precalc_container_arc_inner = precalc_container_arc.clone();
            let receiver_inner = receiver.clone();

            // multiple msm thread can overlap memory copy time and kernel execution time
            std::thread::spawn(move || msm_thread(receiver_inner, precalc_container_arc_inner, i));
        }

        sender
    };
}

fn handle_msm_request(
    kern: &mut MultiexpKernel<GpuG1Affine>,
    request: &MSMRequest,
    _thread_idx: usize,
) -> Result<G1Projective, GPUError> {
    let n = request.n;

    let gpu_result = match request.gpu_op {
        GPU_OP::SETUP_G | GPU_OP::SETUP_SHIFTED_G | GPU_OP::SETUP_LAGRANGE_G | GPU_OP::SETUP_SHIFTED_LAGRANGE_G => {
            let (acc_points, window_size) = kern.setup_ed_base(request.gpu_op).unwrap();

            WINDOW_SIZE.swap(window_size, Ordering::SeqCst);

            let acc_points = unsafe { std::mem::transmute::<_, Vec<G1Projective>>(acc_points) };

            let mut acc_points_lock = ACC_POINTS.write().unwrap();

            match request.gpu_op {
                GPU_OP::SETUP_G => acc_points_lock[0] = acc_points.clone(),
                GPU_OP::SETUP_SHIFTED_G => acc_points_lock[1] = acc_points.clone(),
                GPU_OP::SETUP_LAGRANGE_G => acc_points_lock[2] = acc_points.clone(),
                GPU_OP::SETUP_SHIFTED_LAGRANGE_G => acc_points_lock[3] = acc_points.clone(),
                _ => {}
            }

            GpuG1Projective::zero()
        }

        GPU_OP::LOAD_BASE => match &request.scalars {
            GpuScalars::Slice(scalars) => {
                kern.multiexp(&request.bases, &scalars, request.start, n, request.gpu_op, request.is_mont_form)?
            }
            _ => panic!("LOAD_BASE is not supported!"),
        },

        _ => match &request.scalars {
            GpuScalars::Slice(scalars) => kern.multiexp_precalc_ed(
                &request.bases,
                &scalars,
                request.start,
                n,
                request.gpu_op,
                request.is_mont_form,
            )?,
            GpuScalars::Memory(mem) => kern.multiexp_precalc_ed_gpu_ptr(
                &request.bases,
                mem,
                request.start,
                n,
                request.gpu_op,
                request.is_mont_form,
            )?,
        },
    };

    let snarkvm_result = unsafe { std::mem::transmute::<_, G1Projective>(gpu_result) };

    Ok(snarkvm_result)
}

fn msm_thread(
    input: crossbeam_channel::Receiver<MSMRequest>,
    precalc_container: Arc<MsmPrecalcContainer>,
    thread_idx: usize,
) {
    let gpu_idx: usize = {
        std::env::var("GPU_IDX")
            .and_then(|v| match v.parse() {
                Ok(val) => Ok(val),
                Err(_) => {
                    println!("Invalid env GPU_IDX! Defaulting to 0...");
                    Ok(0)
                }
            })
            .unwrap_or(0)
    };

    let mut kern = MultiexpKernel::<GpuG1Affine>::create(gpu_idx, &precalc_container).unwrap();

    let mut setup_flag = vec![false; 4];
    let mut acc_points_ready_flag = vec![false; 4];

    while let Ok(request) = input.recv() {
        let out = match request.gpu_op {
            GPU_OP::SETUP_G => {
                if setup_flag[0] {
                    Ok(G1Projective::zero())
                } else {
                    setup_flag[0] = true;
                    handle_msm_request(&mut kern, &request, thread_idx).unwrap();

                    Ok(G1Projective::zero())
                }
            }
            GPU_OP::SETUP_SHIFTED_G => {
                if setup_flag[1] {
                    Ok(G1Projective::zero())
                } else {
                    setup_flag[1] = true;
                    handle_msm_request(&mut kern, &request, thread_idx).unwrap();

                    Ok(G1Projective::zero())
                }
            }
            GPU_OP::SETUP_LAGRANGE_G => {
                if setup_flag[2] {
                    Ok(G1Projective::zero())
                } else {
                    setup_flag[2] = true;
                    handle_msm_request(&mut kern, &request, thread_idx).unwrap();

                    Ok(G1Projective::zero())
                }
            }
            GPU_OP::SETUP_SHIFTED_LAGRANGE_G => {
                if setup_flag[3] {
                    Ok(G1Projective::zero())
                } else {
                    setup_flag[3] = true;
                    handle_msm_request(&mut kern, &request, thread_idx).unwrap();

                    Ok(G1Projective::zero())
                }
            }
            GPU_OP::RESET => {
                setup_flag.iter_mut().for_each(|s| *s = false);
                // acc_points.iter_mut().for_each(|s| *s = G1Projective::zero());
                Ok(G1Projective::zero())
            }
            GPU_OP::REUSE_G => {
                if acc_points_ready_flag[0] {
                    handle_msm_request(&mut kern, &request, thread_idx)
                } else {
                    let acc_points_lock = ACC_POINTS.read().unwrap();
                    let window_size = WINDOW_SIZE.load(Ordering::SeqCst);

                    kern.setup_ed_acc_points(request.gpu_op, &(acc_points_lock[0]), window_size).unwrap();

                    acc_points_ready_flag[0] = true;
                    handle_msm_request(&mut kern, &request, thread_idx)
                }
            }
            GPU_OP::REUSE_SHIFTED_G => {
                if acc_points_ready_flag[1] {
                    handle_msm_request(&mut kern, &request, thread_idx)
                } else {
                    let acc_points_lock = ACC_POINTS.read().unwrap();
                    let window_size = WINDOW_SIZE.load(Ordering::SeqCst);

                    kern.setup_ed_acc_points(request.gpu_op, &(acc_points_lock[1]), window_size).unwrap();

                    acc_points_ready_flag[1] = true;
                    handle_msm_request(&mut kern, &request, thread_idx)
                }
            }
            GPU_OP::REUSE_LAGRANGE_G => {
                if acc_points_ready_flag[2] {
                    handle_msm_request(&mut kern, &request, thread_idx)
                } else {
                    let acc_points_lock = ACC_POINTS.read().unwrap();
                    let window_size = WINDOW_SIZE.load(Ordering::SeqCst);

                    kern.setup_ed_acc_points(request.gpu_op, &(acc_points_lock[2]), window_size).unwrap();

                    acc_points_ready_flag[2] = true;
                    handle_msm_request(&mut kern, &request, thread_idx)
                }
            }
            GPU_OP::REUSE_SHIFTED_LAGRANGE_G => {
                if acc_points_ready_flag[3] {
                    handle_msm_request(&mut kern, &request, thread_idx)
                } else {
                    let acc_points_lock = ACC_POINTS.read().unwrap();
                    let window_size = WINDOW_SIZE.load(Ordering::SeqCst);

                    kern.setup_ed_acc_points(request.gpu_op, &(acc_points_lock[3]), window_size).unwrap();

                    acc_points_ready_flag[3] = true;
                    handle_msm_request(&mut kern, &request, thread_idx)
                }
            }
            _ => handle_msm_request(&mut kern, &request, thread_idx),
        };

        request.response.send(out).ok();
    }
}

pub(super) fn msm_gpu<G: AffineCurve, S>(
    mut bases: &[G],
    scalars: &[S],
    start: usize,
    gpu_op: GPU_OP,
    is_mont_form: bool,
) -> Result<G::Projective, GPUError> {
    if TypeId::of::<G>() != TypeId::of::<G1Affine>() {
        return Err(GPUError::Simple("trying to use gpu for unsupported curve".to_owned()));
    }

    if std::mem::size_of::<S>() != std::mem::size_of::<Fr>() {
        return Err(GPUError::Simple("scalar fields have different sizes".to_owned()));
    }

    let mut tscalars = unsafe { std::mem::transmute::<_, &[Fr]>(scalars) };

    if gpu_op != GPU_OP::SETUP_G
        && gpu_op != GPU_OP::SETUP_SHIFTED_G
        && gpu_op != GPU_OP::SETUP_LAGRANGE_G
        && gpu_op != GPU_OP::SETUP_SHIFTED_LAGRANGE_G
    {
        if bases.len() < tscalars.len() {
            tscalars = &tscalars[..bases.len()];
        } else if bases.len() > tscalars.len() {
            bases = &bases[..tscalars.len()];
        }
    }

    let (sender, receiver) = crossbeam_channel::bounded(1);

    MSM_DISPATCH
        .send(MSMRequest {
            bases: unsafe { std::mem::transmute(Arc::new(&bases[..])) },
            scalars: GpuScalars::Slice(unsafe { std::mem::transmute(Arc::new(&tscalars[..])) }),
            response: sender,
            start,
            n: tscalars.len(),
            gpu_op,
            is_mont_form,
        })
        .map_err(|_| GPUError::Simple(format!("sender error: gpu_op = {:?}", gpu_op)))?;

    match receiver.recv() {
        Ok(x) => unsafe { std::mem::transmute_copy(&x) },
        Err(_) => Err(GPUError::Simple(format!("receiver error: gpu_op = {:?}", gpu_op))),
    }
}

pub(super) fn msm_gpu_ptr<G: AffineCurve, Fg>(
    bases: &[G],
    scalars: &DeviceMemory<Fg>,
    start: usize,
    gpu_op: GPU_OP,
    is_mont_form: bool,
) -> Result<G::Projective, GPUError> {
    if TypeId::of::<G>() != TypeId::of::<G1Affine>() {
        return Err(GPUError::Simple("trying to use gpu for unsupported curve".to_owned()));
    }

    assert_eq!(std::mem::size_of::<Fg>(), std::mem::size_of::<GpuFr>());

    let n = std::cmp::min(scalars.size(), bases.len());

    let (sender, receiver) = crossbeam_channel::bounded(1);

    MSM_DISPATCH
        .send(MSMRequest {
            bases: unsafe { std::mem::transmute(Arc::new(&bases[..n])) },
            scalars: GpuScalars::Memory(unsafe { std::mem::transmute::<_, &DeviceMemory<GpuFr>>(scalars) }),
            response: sender,
            start,
            n,
            gpu_op,
            is_mont_form,
        })
        .map_err(|_| GPUError::Simple(format!("sender error: gpu_op = {:?}", gpu_op)))?;

    match receiver.recv() {
        Ok(x) => unsafe { std::mem::transmute_copy(&x) },
        Err(_) => Err(GPUError::Simple(format!("receiver error: gpu_op = {:?}", gpu_op))),
    }
}
