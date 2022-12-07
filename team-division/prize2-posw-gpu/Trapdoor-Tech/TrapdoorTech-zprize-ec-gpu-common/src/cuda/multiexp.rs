use crate::errors::{GPUError, GPUResult};
use crate::structs::{
    AffineCurve, FromBytes, G1Affine, G1Projective, G2Affine, GpuAffine, GpuFr, GpuProjective,
    ProjectiveCurve, Zero, GPU_OP,
};
use crate::{utils::*, CudaFunction};
use crate::{GPUSourceCore, Stream, GPU_CUDA_CORES};
use crate::{GpuEdAffine, GpuEdProjective};

use crypto_cuda::DeviceMemory;

use ark_std::{
    io::{BufReader, Read},
};

use std::any::TypeId;
use std::collections::HashMap;
use std::fs::File;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;

use crate::params::*;
use crate::twisted_edwards::*;

// we transform `T` to arkworks' `G1Affine`
pub fn generate_ed_bases<T>(bases: &[T], name: &str, window_size: usize) {
    use ark_ff::ToBytes;
    use std::io::BufWriter;
    use std::io::Write;

    assert_eq!(std::mem::size_of::<G1Affine>(), std::mem::size_of::<T>());

    let two_addic_window = 1 << window_size;
    let n = bases.len();
    let tbases = unsafe { std::mem::transmute::<_, &[G1Affine]>(bases) };

    let mut precalc_table = vec![GpuEdAffine::default(); n * two_addic_window];

    precalc_table
        .par_chunks_mut(two_addic_window)
        .zip(0..n)
        .for_each(|(b_vec, i)| {
            // make sure z = 1
            let tmp_projective = tbases[i].into_projective();

            let &tmp_projective =
                unsafe { std::mem::transmute::<_, &GpuProjective>(&tmp_projective) };

            let tmp_affine = GpuAffine {
                x: tmp_projective.x,
                y: tmp_projective.y,
            };

            let ed = sw_to_edwards(tmp_affine);
            let ed = edwards_to_neg_one_a(ed);
            let ed = edwards_affine_to_proj(ed);

            let mut acc = GpuEdProjective::default();

            for j in 1..=two_addic_window {
                acc = edwards_add_with_neg_one_a(ed, acc);

                // TODO: we can use batch inversion
                b_vec[j - 1] = edwards_proj_to_affine(acc);
            }
        });

    let mut file = BufWriter::new(File::create(PathBuf::from(name)).unwrap());

    file.write(&n.to_le_bytes()).unwrap();
    file.write(&window_size.to_le_bytes()).unwrap();

    // write the accumulated projective points
    // we have `1 << window_size` points precalculated
    {
        let mut acc = G1Projective::zero();
        let mut acc_points = Vec::new();

        for p in tbases.iter() {
            acc.add_assign_mixed(p);
            acc_points.push(acc.clone());
        }

        acc_points.par_iter_mut().for_each(|p| {
            *p = p.mul(&[two_addic_window as u64]);
        });

        for p in acc_points.iter() {
            p.write(&mut file).unwrap();
        }
    }

    precalc_table
        .iter()
        .for_each(|z| z.write(&mut file).unwrap());
}

pub struct MsmPrecalcContainer {
    // for Twisted Edwards Extended Curves
    ed_powers_of_g_precalc: DeviceMemory<GpuEdAffine>,
    ed_shifted_powers_of_g_precalc: DeviceMemory<GpuEdAffine>,
    ed_lagrange_g_precalc: DeviceMemory<GpuEdAffine>,
    ed_shifted_lagrange_g_precalc: DeviceMemory<GpuEdAffine>,

    window_size: usize,
}

impl MsmPrecalcContainer {
    pub fn create(dev_idx: usize) -> GPUResult<Self> {
        let source_core = &GPU_CUDA_CORES[dev_idx];

        let container = MsmPrecalcContainer::create_with_core(source_core)?;

        Ok(container)
    }

    pub fn create_with_core(core: &GPUSourceCore) -> GPUResult<Self> {
        let core_count = core.device.get_cores()?;
        let mem = core.device.get_memory()?;

        let max_window_size = calc_window_size::<G1Affine>(MAX_SRS_G_LEN, mem, core_count);
        let max_window_num = 1 << max_window_size;

        // for edwards curves
        let ed_powers_of_g_precalc =
            DeviceMemory::<GpuEdAffine>::new(&core.context, MAX_SRS_G_LEN * max_window_num)?;
        let ed_shifted_powers_of_g_precalc =
            DeviceMemory::<GpuEdAffine>::new(&core.context, MAX_SRS_G_LEN * max_window_num)?;
        let ed_lagrange_g_precalc =
            DeviceMemory::<GpuEdAffine>::new(&core.context, MAX_SRS_G_LEN * max_window_num)?;
        let ed_shifted_lagrange_g_precalc =
            DeviceMemory::<GpuEdAffine>::new(&core.context, MAX_SRS_G_LEN * max_window_num)?;

        Ok(MsmPrecalcContainer {
            ed_powers_of_g_precalc,
            ed_shifted_powers_of_g_precalc,
            ed_lagrange_g_precalc,
            ed_shifted_lagrange_g_precalc,

            window_size: max_window_size,
        })
    }

    pub fn get_ed_powers_of_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_powers_of_g_precalc
    }

    pub fn get_ed_shifted_powers_of_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_shifted_powers_of_g_precalc
    }

    pub fn get_ed_lagrange_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_lagrange_g_precalc
    }

    pub fn get_ed_shifted_lagrange_g_precalc(&self) -> &DeviceMemory<GpuEdAffine> {
        &self.ed_shifted_lagrange_g_precalc
    }

    pub fn get_window_size(&self) -> usize {
        self.window_size
    }
}

// Multiexp kernel for a single GPU
pub struct MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    stream: Stream,
    kernel_func_map: Arc<&'a HashMap<String, CudaFunction>>,
    precalc_container: Arc<&'b MsmPrecalcContainer>,

    tmp_base: DeviceMemory<GpuAffine>,
    bucket_buffer: DeviceMemory<GpuProjective>,

    result_buffer: DeviceMemory<GpuProjective>,
    exp_buffer: DeviceMemory<GpuFr>,

    ed_result: DeviceMemory<GpuEdProjective>,
    ed_result_2: DeviceMemory<GpuEdProjective>,

    ed_acc_points_g: Vec<G::Projective>,
    ed_acc_points_shifted_g: Vec<G::Projective>,
    ed_acc_points_lagrange_g: Vec<G::Projective>,
    ed_acc_points_shifted_lagrange_g: Vec<G::Projective>,

    core_count: usize,
    window_size: usize,
    max_window_size: usize,
    max_bucket_size: usize,

    group_name: String,

    _phantom: PhantomData<G>,
}

fn calc_num_groups(core_count: usize, num_windows: usize) -> usize {
    return CORE_N * core_count / num_windows;
}

fn calc_window_size<G>(n: usize, mem_bytes: u64, core_count: usize) -> usize
where
    G: AffineCurve,
{
    let size_affine = std::mem::size_of::<G>();
    let size_projective = std::mem::size_of::<G::Projective>();
    let size_scalar = std::mem::size_of::<G::ScalarField>();

    let size_ed_affine = std::mem::size_of::<GpuEdAffine>();
    let size_ed_projective = std::mem::size_of::<GpuEdProjective>();

    // TODO: not correct
    for i in (MIN_WINDOW_SIZE..=MAX_WINDOW_SIZE).rev() {
        let mem_needed = (n * (2i32.pow(i as u32)) as usize * size_ed_affine) * 4               // 4 ed precalc table
            + n * size_affine                                                                   // 1 bases table
            + CORE_N * core_count * (2i32.pow(MSM_MAX_BUCKET_SIZE as u32)) as usize * size_projective // buckets buffer
            + core_count * CORE_N * size_projective                                             // 1 result buffer
            + 2 * core_count * CORE_N * size_ed_projective                                      // 2 ed result buffer
            + n * size_scalar; // exp buffer

        if (mem_needed as u64) < mem_bytes {
            return i;
        }
    }

    return MIN_WINDOW_SIZE;
}

impl<'a, 'b, G> MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    pub fn setup_ed_acc_points<P>(&mut self, gpu_op: GPU_OP, acc_points: &Vec<P>, window_size: usize) -> GPUResult<()> {
        let acc_points = unsafe { std::mem::transmute::<_, &Vec<G::Projective>>(acc_points) };

        match gpu_op {
            GPU_OP::REUSE_G => {
                self.ed_acc_points_g = (*acc_points).clone();
            }
            GPU_OP::REUSE_SHIFTED_G => {
                self.ed_acc_points_shifted_g = (*acc_points).clone();
            }
            GPU_OP::REUSE_LAGRANGE_G => {
                self.ed_acc_points_lagrange_g = (*acc_points).clone();
            }
            GPU_OP::REUSE_SHIFTED_LAGRANGE_G => {
                self.ed_acc_points_shifted_lagrange_g = (*acc_points).clone();
            }
            _ => panic!("invalid gpu_op"),
        }

        self.window_size = window_size;

        Ok(())
    }

    pub fn setup_ed_base(&mut self, gpu_op: GPU_OP) -> GPUResult<(Vec<G::Projective>, usize)> {
        let mut acc_points = Vec::new();

        let bases_precalc = match gpu_op {
            GPU_OP::SETUP_G
            | GPU_OP::SETUP_SHIFTED_G
            | GPU_OP::SETUP_LAGRANGE_G
            | GPU_OP::SETUP_SHIFTED_LAGRANGE_G => {
                let mut bases_precalc = Vec::new();
                let mut file_reader = if gpu_op == GPU_OP::SETUP_G {
                    BufReader::new(
                        File::open(PathBuf::from("ed_g_prepared.dat"))
                            .map_err(|_| GPUError::Simple("Cannot read ed_g".to_owned()))?,
                    )
                } else if gpu_op == GPU_OP::SETUP_SHIFTED_G {
                    BufReader::new(
                        File::open(PathBuf::from("ed_shifted_g_prepared.dat".to_owned()))
                            .map_err(|_| GPUError::Simple("Cannot read ed_shifted_g".to_owned()))?,
                    )
                } else if gpu_op == GPU_OP::SETUP_LAGRANGE_G {
                    BufReader::new(
                        File::open(PathBuf::from("ed_lagrange_g_prepared.dat".to_owned()))
                            .map_err(|_| {
                                GPUError::Simple("Cannot read ed_lagrange_g".to_owned())
                            })?,
                    )
                } else {
                    BufReader::new(
                        File::open(PathBuf::from(
                            "ed_shifted_lagrange_g_prepared.dat".to_owned(),
                        ))
                        .map_err(|_| {
                            GPUError::Simple("Cannot read ed_shifted_lagrange_g".to_owned())
                        })?,
                    )
                };

                let mut buffer = [0u8; 8];
                file_reader.read(&mut buffer[..]).unwrap();
                let n = usize::from_le_bytes(buffer);
                file_reader.read(&mut buffer[..]).unwrap();
                let window_size = usize::from_le_bytes(buffer);

                if n > MAX_SRS_G_LEN {
                    return Err(GPUError::Simple(format!(
                        "n = {n} is not supported during setup"
                    )));
                }

                if window_size > self.max_window_size {
                    return Err(GPUError::Simple(format!(
                        "window_size = {window_size} is not supported during setup"
                    )));
                }

                for _ in 0..n {
                    let acc_point = G::Projective::read(&mut file_reader).unwrap();

                    acc_points.push(acc_point);
                }

                //by using NAF we can increase our `window_size` by 1 bit
                self.window_size = window_size + 1;

                loop {
                    let power: std::io::Result<GpuEdAffine> = FromBytes::read(&mut file_reader);

                    if let Ok(power) = power {
                        bases_precalc.push(power.clone());
                    } else {
                        break;
                    }
                }

                bases_precalc
            }

            _ => {
                return Err(GPUError::Simple(
                    "Invalid GPU_OP during base setup".to_owned(),
                ))
            }
        };

        let tbases_precalc = unsafe { &*(bases_precalc.as_ref() as *const [GpuEdAffine]) };

        let memory = match gpu_op {
            GPU_OP::SETUP_G => self.precalc_container.get_ed_powers_of_g_precalc(),
            GPU_OP::SETUP_SHIFTED_G => self.precalc_container.get_ed_shifted_powers_of_g_precalc(),
            GPU_OP::SETUP_LAGRANGE_G => self.precalc_container.get_ed_lagrange_g_precalc(),
            GPU_OP::SETUP_SHIFTED_LAGRANGE_G => {
                self.precalc_container.get_ed_shifted_lagrange_g_precalc()
            }
            _ => {
                return Err(GPUError::Simple(
                    "Invalid GPU_OP during base setup".to_owned(),
                ))
            }
        };

        memory.read_from_async(tbases_precalc, bases_precalc.len(), &self.stream)?;

        self.stream.sync()?;

        Ok((acc_points, self.window_size))
    }

    pub fn create_with_core(
        core: &'a GPUSourceCore,
        container: &'b MsmPrecalcContainer,
    ) -> GPUResult<MultiexpKernel<'a, 'b, G>> {
        let kernel_func_map = Arc::new(&core.kernel_func_map);
        let precalc_container = Arc::new(container);

        let core_count = core.device.get_cores()?;
        let mem = core.device.get_memory()?;

        let max_window_size = calc_window_size::<G>(MAX_SRS_G_LEN, mem, core_count);
        let max_bucket_size = MSM_MAX_BUCKET_SIZE;

        let n = MAX_SRS_G_LEN;
        let max_bucket_num = 1 << max_bucket_size;

        let exp_buf = DeviceMemory::<GpuFr>::new(&core.context, n)?;

        // keep these two buffers in case we need pippenger algorithm
        let tmp_base = DeviceMemory::<GpuAffine>::new(&core.context, n)?;
        let buck_buf = DeviceMemory::<GpuProjective>::new(
            &core.context,
            CORE_N * core_count * max_bucket_num,
        )?;
        let res_buf = DeviceMemory::<GpuProjective>::new(&core.context, CORE_N * core_count)?;

        let ed_result = DeviceMemory::<GpuEdProjective>::new(&core.context, CORE_N * core_count)?;
        let ed_result_2 = DeviceMemory::<GpuEdProjective>::new(&core.context, CORE_N * core_count)?;

        let ed_acc_points_g = Vec::new();
        let ed_acc_points_shifted_g = Vec::new();
        let ed_acc_points_lagrange_g = Vec::new();
        let ed_acc_points_shifted_lagrange_g = Vec::new();

        let group_name = if TypeId::of::<G>() == TypeId::of::<G1Affine>() {
            String::from("G1")
        } else if TypeId::of::<G>() == TypeId::of::<G2Affine>() {
            String::from("G2")                                                  // G2 MSM is not used
        } else {
            panic!("not supported elliptic curves!")
        };

        let stream = Stream::new_with_context(&core.context)?;

        Ok(MultiexpKernel {
            stream,
            kernel_func_map,
            precalc_container,

            tmp_base,

            bucket_buffer: buck_buf,
            result_buffer: res_buf,
            exp_buffer: exp_buf,

            ed_result,
            ed_result_2,

            ed_acc_points_g,
            ed_acc_points_shifted_g,
            ed_acc_points_lagrange_g,
            ed_acc_points_shifted_lagrange_g,

            core_count,
            window_size: max_window_size,
            max_window_size,
            max_bucket_size,

            group_name,

            _phantom: PhantomData,
        })
    }

    pub fn create(
        dev_idx: usize,
        container: &'b MsmPrecalcContainer,
    ) -> GPUResult<MultiexpKernel<'a, 'b, G>> {
        let source_core = &GPU_CUDA_CORES[dev_idx];

        let kernel = MultiexpKernel::create_with_core(source_core, container)?;

        Ok(kernel)
    }

    /// naive multiexp, use pippenger algorithm
    pub fn multiexp<T, TS>(
        &mut self,
        bases: &[T],
        exps: &[TS],
        start: usize,
        n: usize,
        gpu_op: GPU_OP,
        is_mont_form: bool,
    ) -> GPUResult<G::Projective> {
        let bases = unsafe { std::mem::transmute::<_, &[G]>(bases) };
        let exps = unsafe { std::mem::transmute::<_, &[G::ScalarField]>(exps) };

        let exp_bits = std::mem::size_of::<G::ScalarField>() * 8;

        // `window_size` is designated by `bucket_size`
        let window_size = self.max_bucket_size;

        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = std::cmp::min(n, calc_num_groups(self.core_count / 2, num_windows));

        let texps = unsafe { std::mem::transmute::<&[G::ScalarField], &[GpuFr]>(exps) };

        let mut total = num_windows * num_groups;
        total += (LOCAL_WORK_SIZE - (total % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let (gws, lws) = calc_cuda_wg_threads(total);

        // prepare exps in non-montgomery form
        let exp_buf = self.exp_buffer.get_inner();
        self.exp_buffer
            .read_from_async(texps, exps.len(), &self.stream)?;

        if is_mont_form {
            let (gws, lws) = calc_cuda_wg_threads(n);
            let f = self.kernel_func_map.get("Fr_poly_unmont").unwrap().inner;

            let params = make_params!(exp_buf, n as u32);
            let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

            self.stream.launch(&kern)?;
        }

        if gpu_op == GPU_OP::LOAD_BASE {
            // GpuAffine does not have `inf` field, so we need to do a transform `G` -> `GpuAffine`
            let mut tbases = vec![GpuAffine::default(); bases.len()];
            for i in 0..bases.len() {
                // make sure z = 1
                let tmp_projective = bases[i].into_projective();

                let &tmp = unsafe { std::mem::transmute::<_, &GpuProjective>(&tmp_projective) };
                tbases[i].x = tmp.x;
                tbases[i].y = tmp.y;
            }

            self.tmp_base
                .read_from_async(&tbases, bases.len(), &self.stream)?;
        }

        let base_buf = match gpu_op {
            GPU_OP::LOAD_BASE => self.tmp_base.get_inner(),
            _ => panic!("invalid gpu_op"),
        };

        let base_start = if gpu_op == GPU_OP::LOAD_BASE {
            0 as u32
        } else {
            start as u32
        };

        let buck_buf = self.bucket_buffer.get_inner();
        let res_buf = self.result_buffer.get_inner();
        let exp_buf = self.exp_buffer.get_inner();

        let kernel_name = format!("{}_bellman_multiexp", self.group_name);
        let f = self.kernel_func_map.get(&kernel_name).unwrap().inner;

        let params = make_params!(
            base_buf,
            buck_buf,
            res_buf,
            exp_buf,
            base_start as u32,
            n as u32,
            num_groups as u32,
            num_windows as u32,
            window_size as u32
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        let (gws, lws) = (num_windows, CUDA_LWS);

        let kernel_name = format!("{}_group_acc", self.group_name);
        let f = self.kernel_func_map.get(&kernel_name).unwrap().inner;

        let params = make_params!(res_buf, num_groups as u32, num_windows as u32, lws as u32);
        let kern = create_kernel_with_params!(f, <<<gws as u32, lws as u32, 0>>>(params));

        self.stream.launch(&kern)?;

        let mut res = vec![G::Projective::zero(); num_windows];

        let tres =
            unsafe { &mut *(res.as_mut_slice() as *mut [G::Projective] as *mut [GpuProjective]) };

        self.result_buffer
            .write_to_async(tres, res.len(), &self.stream)?;

        self.stream.sync()?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = G::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double_in_place();
            }

            acc += &res[i];
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

impl<'a, 'b, G> MultiexpKernel<'a, 'b, G>
where
    G: AffineCurve,
{
    pub fn multiexp_precalc_ed<T, TS>(
        &mut self,
        _bases: &[T],
        exps: &[TS],
        start: usize,
        n: usize,
        gpu_op: GPU_OP,
        is_mont_form: bool,
    ) -> GPUResult<G::Projective> {
        let texps = unsafe { std::mem::transmute::<_, &[GpuFr]>(exps) };

        // prepare exps in non-montgomery form
        self.exp_buffer
            .read_from_async(texps, texps.len(), &self.stream)?;

        let res = self.multiexp_precalc_ed_core(start, n, gpu_op, is_mont_form)?;

        Ok(res)
    }

    pub fn multiexp_precalc_ed_gpu_ptr<T>(
        &mut self,
        _bases: &[T],
        gpu_mem: &DeviceMemory<GpuFr>,
        start: usize,
        n: usize,
        gpu_op: GPU_OP,
        is_mont_form: bool,
    ) -> GPUResult<G::Projective> {
        // prepare exps in non-montgomery form
        DeviceMemory::<GpuFr>::memcpy_from_to_async(
            gpu_mem,
            &self.exp_buffer,
            Some(n),
            &self.stream,
        )?;

        let res = self.multiexp_precalc_ed_core(start, n, gpu_op, is_mont_form)?;

        Ok(res)
    }

    /// use precalculated table to accelerate MSM
    /// assume exp data is ready in self.exp_buffer
    pub fn multiexp_precalc_ed_core(
        &mut self,
        start: usize,
        n: usize,
        gpu_op: GPU_OP,
        is_mont_form: bool,
    ) -> GPUResult<G::Projective> {
        // does not support LOAD_BASE mode, since the recalculation of bases table would cost too much
        if gpu_op == GPU_OP::LOAD_BASE {
            panic!("gpu_op {:?} is not supported!", gpu_op);
        }

        let exp_bits = std::mem::size_of::<G::ScalarField>() * 8;

        let use_naf = true;

        let window_size = if use_naf {
            self.window_size
        } else {
            self.window_size - 1
        };

        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = std::cmp::min(n, calc_num_groups(self.core_count, num_windows));

        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let mut total = num_windows * num_groups;
        total += (LOCAL_WORK_SIZE - (total % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let exp_buf = self.exp_buffer.get_inner();
        let res_buf = self.ed_result.get_inner();

        let (gws, lws) = calc_cuda_wg_threads(total);

        if is_mont_form {
            let (gws, lws) = calc_cuda_wg_threads(n);
            let f = self.kernel_func_map.get("Fr_poly_unmont").unwrap().inner;

            let params = make_params!(exp_buf, n as u32);
            let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

            self.stream.launch(&kern)?;
        }

        let kernel_name = if use_naf {
            format!("{}_multiexp_ed_neg_one_a_precalc_naf", self.group_name)
        } else {
            format!("{}_multiexp_ed_neg_one_a_precalc", self.group_name)
        };

        let f = self.kernel_func_map.get(&kernel_name).unwrap().inner;

        let base_precalc_buf = match gpu_op {
            GPU_OP::REUSE_G => self
                .precalc_container
                .get_ed_powers_of_g_precalc()
                .get_inner(),
            GPU_OP::REUSE_SHIFTED_G => self
                .precalc_container
                .get_ed_shifted_powers_of_g_precalc()
                .get_inner(),
            GPU_OP::REUSE_LAGRANGE_G => self
                .precalc_container
                .get_ed_lagrange_g_precalc()
                .get_inner(),
            GPU_OP::REUSE_SHIFTED_LAGRANGE_G => self
                .precalc_container
                .get_ed_shifted_lagrange_g_precalc()
                .get_inner(),
            _ => panic!("invalid gpu_op"),
        };

        let acc_points = match gpu_op {
            GPU_OP::REUSE_G => &self.ed_acc_points_g,
            GPU_OP::REUSE_SHIFTED_G => &self.ed_acc_points_shifted_g,
            GPU_OP::REUSE_LAGRANGE_G => &self.ed_acc_points_lagrange_g,
            GPU_OP::REUSE_SHIFTED_LAGRANGE_G => &self.ed_acc_points_shifted_lagrange_g,
            _ => panic!("invalid gpu_op"),
        };

        let base_start = if gpu_op == GPU_OP::LOAD_BASE {
            0 as u32
        } else {
            start as u32
        };


        let params = make_params!(
            base_precalc_buf,
            res_buf,
            exp_buf,
            base_start as u32,
            n as u32,
            num_groups as u32,
            num_windows as u32,
            window_size as u32
        );
        let kern = create_kernel_with_params!(f, <<<gws, lws, 0>>>(params));

        self.stream.launch(&kern)?;

        let mut in_src = true;
        let mut remaining_groups = num_groups as u32;
        let max_p = 4;

        // empiric value
        while remaining_groups > max_p {
            let src_buf = if in_src {
                self.ed_result.get_inner()
            } else {
                self.ed_result_2.get_inner()
            };

            let dst_buf = if in_src {
                self.ed_result_2.get_inner()
            } else {
                self.ed_result.get_inner()
            };

            // maybe we can do better
            // we use a bunch of threads to compute group_acc inside one window, each thread process maximum of `max_p` point add
            // then we scale it to `num_windows`
            let total = ((remaining_groups) / max_p + ((remaining_groups) % max_p != 0) as u32)
                * num_windows as u32;
            let (gws, lws) = calc_cuda_wg_threads(total as usize);

            let kernel_name = format!("{}_multiexp_ed_neg_one_a_group_acc_iter", self.group_name);
            let f = self.kernel_func_map.get(&kernel_name).unwrap().inner;

            let params = make_params!(
                src_buf,
                dst_buf,
                total,
                remaining_groups,
                num_windows as u32,
                max_p as u32
            );
            let kern = create_kernel_with_params!(f, <<<gws as u32, lws as u32, 0>>>(params));

            self.stream.launch(&kern)?;

            remaining_groups = remaining_groups / max_p + ((remaining_groups) % max_p != 0) as u32;

            in_src = !in_src;
        }

        let mut res = vec![GpuEdProjective::default(); num_windows * remaining_groups as usize];
        let len = res.len();

        if in_src {
            self.ed_result.write_to_async(&mut res, len, &self.stream)?;
        } else {
            self.ed_result_2
                .write_to_async(&mut res, len, &self.stream)?;
        }
        self.stream.sync()?;

        let ed_acc_point = if use_naf {
            // make sure z = 1
            let acc_point = if start == 0 {
                acc_points[n - 1].into_affine().into_projective()
            } else {
                let tmp_point = acc_points[n + start - 1] - acc_points[start - 1];
                tmp_point.into_affine().into_projective()
            };

            let &tmp_projective = unsafe { std::mem::transmute::<_, &GpuProjective>(&acc_point) };

            let tmp_affine = GpuAffine {
                x: tmp_projective.x,
                y: tmp_projective.y,
            };

            let ed = sw_to_edwards(tmp_affine);
            let ed = edwards_to_neg_one_a(ed);
            let ed = edwards_affine_to_proj(ed);

            ed
        } else {
            GpuEdProjective::default()
        };

        let real_res = res
            .chunks(remaining_groups as usize)
            .map(|slice| {
                let mut acc = GpuEdProjective::default();
                for x in slice.iter() {
                    acc = edwards_add_with_neg_one_a(acc, *x);
                }

                if use_naf {
                    acc = edwards_add_with_neg_one_a(acc, ed_acc_point);
                }

                acc
            })
            .collect::<Vec<_>>();

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = GpuEdProjective::default();
        let mut bits = 0;

        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc = edwards_double_with_neg_one_a(acc);
            }

            acc = edwards_add_with_neg_one_a(acc, real_res[i]);
            bits += w; // Process the next window
        }

        let res = edwards_to_sw_proj(edwards_from_neg_one_a(edwards_proj_to_affine(acc)));

        let &res = unsafe { std::mem::transmute::<_, &G::Projective>(&res) };

        Ok(res)
    }
}
