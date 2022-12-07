#[cfg(test)]
mod test_buffer_as_poly {
    use crate::GPU_CUDA_CORES;

    use crate::cuda::{GpuPoly, PolyKernel};
    use crate::structs::*;
    use crate::{log2_floor, GPUResult};

    use ark_bls12_377::Fr;
    use ark_ff::FftField;
    use ark_poly::{
        domain::EvaluationDomain, domain::GeneralEvaluationDomain, univariate::DensePolynomial,
        Evaluations, Polynomial, UVPolynomial,
    };
    use ark_std::{One, UniformRand, Zero};

    use std::ops::{AddAssign, Mul, MulAssign, SubAssign};

    use log::info;

    const TEST_VEC_LEN: usize = 1 << 16;
    const GPU_DEV_IDX: usize = 0;

    fn init() {
        let _ = env_logger::builder().try_init();
    }

    #[test]
    #[serial]
    pub fn test_buffer_as_poly_weird() -> GPUResult<()> {
        init();

        let mut rng = ark_std::test_rng();

        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX])?;

        let test_vec = (0..TEST_VEC_LEN)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut gpu_test_vec = vec![Fr::one(); TEST_VEC_LEN];

        info!("testing polynomial arithmetic - weird [gpu]");

        let mut gpu_poly = GpuPoly::new(&poly_kern, TEST_VEC_LEN)?;

        gpu_poly.fill_with_fe(&Fr::zero())?;
        gpu_poly.read_from(&test_vec)?;
        gpu_poly.write_to(&mut gpu_test_vec)?;

        for i in 0..gpu_test_vec.len() {
            if gpu_test_vec[i] != test_vec[i] {
                println!(
                    "gpu_test_vec[{}] = {}, test_vec[{}] = {}",
                    i, gpu_test_vec[i], i, test_vec[i]
                );
            }
        }

        Ok(())
    }

    #[test]
    #[serial]
    pub fn test_buffer_as_poly_arith_scalar() -> GPUResult<()> {
        init();

        let mut rng = ark_std::test_rng();

        let test_fr = Fr::rand(&mut rng);
        let test_fr_vec = vec![test_fr; TEST_VEC_LEN];

        let domain: GeneralEvaluationDomain<Fr> = EvaluationDomain::new(TEST_VEC_LEN).unwrap();

        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX])?;

        let test_vec = (0..TEST_VEC_LEN)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut gpu_test_vec = test_vec.clone();

        info!("testing polynomial arithmetic - scalar [gpu]");

        let test = DensePolynomial::from_coefficients_vec(test_vec.clone());

        println!(
            "test.len = {}, test_fr_evals.len = {}",
            test.coeffs.len(),
            test_fr_vec.len()
        );

        info!("testing polynomial arithmetic - scalar - scale");
        let new_result = test.mul(test_fr);

        let mut gpu_poly = GpuPoly::new(&poly_kern, TEST_VEC_LEN)?;

        gpu_poly.read_from(&gpu_test_vec[..])?;
        gpu_poly.scale(&test_fr)?;
        gpu_poly.write_to(&mut gpu_test_vec[..])?;

        assert_eq!(new_result.coeffs(), gpu_test_vec);

        info!("testing polynomial arithmetic - scalar - add_constant");

        let mut gpu_test_vec = test_vec.clone();
        let mut test = Evaluations::from_vec_and_domain(test_vec.clone(), domain);
        let test_fr_evals = Evaluations::from_vec_and_domain(test_fr_vec.clone(), domain);

        println!(
            "test.len = {}, test_fr_evals.len = {}",
            test.evals.len(),
            test_fr_evals.evals.len()
        );

        test += &test_fr_evals;

        gpu_poly.read_from(&gpu_test_vec)?;
        gpu_poly.add_constant(&test_fr)?;
        gpu_poly.write_to(&mut gpu_test_vec)?;

        assert_eq!(test.evals, gpu_test_vec);

        info!("testing polynomial arithmetic - scalar - sub_constant");

        let mut gpu_test_vec = test_vec.clone();
        let mut test = Evaluations::from_vec_and_domain(test_vec.clone(), domain);
        let test_fr_evals = Evaluations::from_vec_and_domain(test_fr_vec.clone(), domain);

        println!(
            "test.len = {}, test_fr_evals.len = {}",
            test.evals.len(),
            test_fr_evals.evals.len()
        );

        test -= &test_fr_evals;

        gpu_poly.read_from(&gpu_test_vec)?;
        gpu_poly.sub_constant(&test_fr)?;
        gpu_poly.write_to(&mut gpu_test_vec)?;

        assert_eq!(test.evals, gpu_test_vec);

        Ok(())
    }

    #[test]
    #[serial]
    pub fn test_buffer_as_poly_arith_vector() -> GPUResult<()> {
        init();

        let mut rng = ark_std::test_rng();

        let domain: GeneralEvaluationDomain<Fr> = EvaluationDomain::new(TEST_VEC_LEN).unwrap();

        let test_fr = Fr::rand(&mut rng);
        let test_fr_vec = vec![test_fr; TEST_VEC_LEN];
        let test_fr_evals = Evaluations::from_vec_and_domain(test_fr_vec.clone(), domain);

        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX])?;

        let mut gpu_poly_1 = GpuPoly::<Fr>::new(&poly_kern, TEST_VEC_LEN)?;
        let mut gpu_poly_2 = GpuPoly::<Fr>::new(&poly_kern, TEST_VEC_LEN)?;

        let test_vec = (0..TEST_VEC_LEN)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        info!("testing polynomial arithmetic - vectors [gpu]");

        let mut gpu_test_vec = test_vec.clone();
        let gpu_test_vec_2 = test_vec.clone();

        let mut test = Evaluations::from_vec_and_domain(test_vec.clone(), domain);
        let test_2 = test.clone();

        info!("testing polynomial arithmetic - vectors - add_assign");
        test.add_assign(&test_2);

        gpu_poly_1.read_from(&gpu_test_vec)?;
        gpu_poly_2.read_from(&gpu_test_vec_2)?;
        gpu_poly_1.add_assign(&gpu_poly_2)?;
        gpu_poly_1.write_to(&mut gpu_test_vec[..])?;

        assert_eq!(test.evals, gpu_test_vec);

        let mut gpu_test_vec = test_vec.clone();
        let gpu_test_vec_2 = test_vec.clone();

        let mut test = Evaluations::from_vec_and_domain(test_vec.clone(), domain);
        let test_2 = test.clone();

        info!("testing polynomial arithmetic - vectors - sub_assign");
        test.sub_assign(&test_2);

        gpu_poly_1.read_from(&gpu_test_vec)?;
        gpu_poly_2.read_from(&gpu_test_vec_2)?;
        gpu_poly_1.sub_assign(&gpu_poly_2)?;
        gpu_poly_1.write_to(&mut gpu_test_vec[..])?;

        assert_eq!(test.evals, gpu_test_vec);

        let mut gpu_test_vec = test_vec.clone();
        let gpu_test_vec_2 = test_vec.clone();

        let mut test = Evaluations::from_vec_and_domain(test_vec.clone(), domain);
        let test_2 = test.clone();

        info!("testing polynomial arithmetic - vectors - mul_assign");
        test.mul_assign(&test_2);

        gpu_poly_1.read_from(&gpu_test_vec)?;
        gpu_poly_2.read_from(&gpu_test_vec_2)?;
        gpu_poly_1.mul_assign(&gpu_poly_2)?;
        gpu_poly_1.write_to(&mut gpu_test_vec[..])?;

        assert_eq!(test.evals, gpu_test_vec);

        let mut gpu_test_vec = test_vec.clone();
        let gpu_test_vec_2 = test_vec.clone();

        let mut test = Evaluations::from_vec_and_domain(test_vec.clone(), domain);
        let mut test_2 = test.clone();

        info!("testing polynomial arithmetic - vectors - add_assign_scaled");
        test_2.mul_assign(&test_fr_evals);
        test.add_assign(&test_2);

        gpu_poly_1.read_from(&gpu_test_vec)?;
        gpu_poly_2.read_from(&gpu_test_vec_2)?;
        gpu_poly_1.add_assign_scale(&gpu_poly_2, &test_fr)?;
        gpu_poly_1.write_to(&mut gpu_test_vec[..])?;

        assert_eq!(test.evals, gpu_test_vec);

        let mut gpu_test_vec = test_vec.clone();
        let gpu_test_vec_2 = test_vec.clone();

        let mut test = Evaluations::from_vec_and_domain(test_vec.clone(), domain);
        let mut test_2 = test.clone();

        info!("testing polynomial arithmetic - vectors - sub_assign_scaled");
        test_2.mul_assign(&test_fr_evals);
        test.sub_assign(&test_2);

        gpu_poly_1.read_from(&gpu_test_vec)?;
        gpu_poly_2.read_from(&gpu_test_vec_2)?;
        gpu_poly_1.sub_assign_scale(&gpu_poly_2, &test_fr)?;
        gpu_poly_1.write_to(&mut gpu_test_vec[..])?;

        assert_eq!(test.evals, gpu_test_vec);

        Ok(())
    }

    #[test]
    #[serial]
    pub fn test_buffer_as_poly_arith_reductions() -> GPUResult<()> {
        init();

        let mut rng = ark_std::test_rng();

        let test_fr = Fr::rand(&mut rng);

        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX])?;

        let mut gpu_poly = GpuPoly::<Fr>::new(&poly_kern, TEST_VEC_LEN)?;
        let mut gpu_powers = GpuPoly::<Fr>::new_powers(&poly_kern)?;
        let mut gpu_results = GpuPoly::<Fr>::new(&poly_kern, poly_kern.get_core_x_batch())?;

        let test_vec = (0..TEST_VEC_LEN)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let gpu_test_vec = test_vec.clone();

        info!("testing polynomial arithmetic - reductions [gpu]");

        let test = DensePolynomial::from_coefficients_vec(test_vec.clone());

        info!("testing polynomial arithmetic - reductions - evaluate_at");
        let cpu_res = test.evaluate(&test_fr);

        gpu_poly.read_from(&gpu_test_vec)?;
        let gpu_res = gpu_poly.evaluate_at(&mut gpu_powers, &mut gpu_results, &test_fr)?;

        assert_eq!(cpu_res, gpu_res);

        Ok(())
    }

    #[test]
    #[serial]
    pub fn test_buffer_as_poly_arith_mappings() -> GPUResult<()> {
        init();

        let mut rng = ark_std::test_rng();

        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX])?;

        let mut gpu_poly = GpuPoly::<Fr>::new(&poly_kern, TEST_VEC_LEN)?;
        let mut gpu_powers = GpuPoly::new_powers(&poly_kern)?;

        let test_vec = (0..TEST_VEC_LEN)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        info!("testing polynomial arithmetic - mappings [gpu]");

        let mut test = DensePolynomial::from_coefficients_vec(test_vec.clone());
        let mut gpu_test_vec = test_vec.clone();

        info!("testing polynomial arithmetic - mappings - batch_inversion");

        test.coeffs.iter_mut().for_each(|g| {
            g.inverse_in_place().unwrap();
        });

        gpu_poly.read_from(&gpu_test_vec)?;
        gpu_poly.batch_inversion_full()?;
        gpu_poly.write_to(&mut gpu_test_vec)?;

        let coeffs = test.coeffs.to_vec();

        assert_eq!(coeffs, gpu_test_vec);

        info!("testing polynomial arithmetic - mappings - distribute_powers");

        let mut test = DensePolynomial::from_coefficients_vec(test_vec.clone());
        let mut gpu_test_vec = test_vec.clone();

        let g = Fr::rand(&mut rng);
        let mut accumulate_g = Fr::one();

        test.coeffs.iter_mut().for_each(|f| {
            *f = *f * accumulate_g;
            accumulate_g = accumulate_g * g;
        });

        gpu_poly.read_from(&gpu_test_vec)?;
        gpu_poly.distribute_powers(&mut gpu_powers, &g, 0)?;
        gpu_poly.write_to(&mut gpu_test_vec)?;

        let coeffs = test.coeffs.to_vec();

        assert_eq!(coeffs, gpu_test_vec);

        Ok(())
    }

    #[test]
    #[serial]
    pub fn test_buffer_as_poly_ntt() -> GPUResult<()> {
        init();

        info!("testing polynomial arithmetic - fft");

        let max_size = TEST_VEC_LEN;

        let deg = log2_floor(max_size);

        let sizes = 1 << deg;

        let mut rng = ark_std::test_rng();
        let domain: GeneralEvaluationDomain<Fr> = GeneralEvaluationDomain::new(sizes).unwrap();

        let scalars = (0..sizes).map(|_| Fr::rand(&mut rng)).collect::<Vec<_>>();

        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX])?;

        let mut gpu_poly = GpuPoly::new(&poly_kern, sizes)?;

        let mut coeffs_0 = scalars[..sizes].to_vec().clone();
        let mut coeffs_1 = coeffs_0.clone();

        let real_omega = Fr::get_root_of_unity(sizes).unwrap();
        let real_omega_inv = real_omega.inverse().unwrap();

        let time = std::time::Instant::now();

        info!("testing polynomial arithmetic - fft performance - [cpu]");
        domain.fft_in_place(&mut coeffs_0);

        info!("cpu fft time = {:?}", time.elapsed());

        let time = std::time::Instant::now();

        info!("testing polynomial arithmetic - fft performance - [gpu]");

        gpu_poly.read_from(&coeffs_1)?;
        gpu_poly.fft_full(&real_omega)?;
        gpu_poly.write_to(&mut coeffs_1)?;

        info!("gpu fft time = {:?}", time.elapsed());

        assert_eq!(coeffs_1, coeffs_0);

        info!("testing polynomial arithmetic - ifft");

        let mut coeffs_0 = scalars[..sizes].to_vec().clone();
        let mut coeffs_1 = coeffs_0.clone();

        let time = std::time::Instant::now();

        info!("testing polynomial arithmetic - ifft performance - [cpu]");
        domain.ifft_in_place(&mut coeffs_0);

        info!("cpu ifft time = {:?}", time.elapsed());

        let time = std::time::Instant::now();

        info!("testing polynomial arithmetic - ifft performance - [gpu]");

        gpu_poly.read_from(&coeffs_1)?;
        gpu_poly.ifft_full(&real_omega_inv)?;
        gpu_poly.write_to(&mut coeffs_1)?;

        info!("gpu ifft time = {:?}", time.elapsed());

        assert_eq!(coeffs_1, coeffs_0);

        Ok(())
    }
}

#[cfg(test)]
mod test_multiexp {
    use crate::cuda::*;
    use crate::structs::*;
    use crate::GPU_CUDA_CORES;
    use ark_bls12_377::{Fr, G1Affine, G1Projective};
    use ark_ec::ProjectiveCurve;
    use ark_ff::fields::PrimeField;
    use ark_std::rand::Rng;
    use ark_std::UniformRand;

    use ark_ec::msm::VariableBaseMSM;

    const MAX_LENGTH: usize = 1 << 16;
    const GPU_DEV_IDX: usize = 0;

    use log::info;

    #[test]
    #[serial]
    pub fn test_gpu_multiexp() {
        let _ = env_logger::try_init();

        use std::time::Instant;

        let mut fr_rng = ark_std::test_rng();
        let mut rng = ark_std::test_rng();

        info!("n = {}", MAX_LENGTH);
        info!("start generating fr");
        let v = (0..MAX_LENGTH).map(|_| fr_rng.gen()).collect::<Vec<Fr>>();

        info!("start generating affine points");
        let g = (0..MAX_LENGTH)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect::<Vec<G1Affine>>();

        let cpu_start = Instant::now();
        let start = Instant::now();
        let v1 = v.iter().map(|v| v.into_repr()).collect::<Vec<_>>();
        info!("cpu into_repr takes {:?}", start.elapsed());

        let start = Instant::now();
        let cpu = VariableBaseMSM::multi_scalar_mul(&g, &v1);
        info!("cpu = {:?}", cpu);
        info!("cpu msm takes {:?}", start.elapsed());
        info!("cpu total takes {:?}", cpu_start.elapsed());

        let precalc_container =
            MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX]).unwrap();
        let mut kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();

        let start = Instant::now();

        let gpu = kern
            .multiexp(&g, &v, 0, MAX_LENGTH, GPU_OP::LOAD_BASE, true)
            .unwrap();

        info!("gpu msm takes {:?}", start.elapsed());
        assert_eq!(cpu.into_affine(), gpu.into_affine());

        let start = Instant::now();
        let gpu = kern
            .multiexp(&g, &v1, 0, MAX_LENGTH, GPU_OP::LOAD_BASE, false)
            .unwrap();

        info!("gpu msm takes {:?}", start.elapsed());
        assert_eq!(cpu.into_affine(), gpu.into_affine());
    }

    // 2022.07.27: since we do not use SW curves any more, these tests should be disabled
    // #[test]
    // #[serial]
    // pub fn test_gpu_precalc_multiexp() {
    //     let _ = env_logger::try_init();

    //     use std::time::Instant;

    //     let mut fr_rng = ark_std::test_rng();
    //     let mut rng = ark_std::test_rng();

    //     info!("n = {}", MAX_LENGTH);
    //     info!("start generating fr");
    //     let v = (0..MAX_LENGTH).map(|_| fr_rng.gen()).collect::<Vec<Fr>>();

    //     info!("start generating affine points");
    //     let g = (0..MAX_LENGTH)
    //         .map(|_| G1Projective::rand(&mut rng).into_affine())
    //         .collect::<Vec<G1Affine>>();

    //     let cpu_start = Instant::now();
    //     let start = Instant::now();
    //     let v1 = v.iter().map(|v| v.into_repr()).collect::<Vec<_>>();

    //     info!("cpu into_repr takes {:?}", start.elapsed());

    //     let start = Instant::now();
    //     let cpu = VariableBaseMSM::multi_scalar_mul(&g, &v1);

    //     info!("cpu = {:?}", cpu);
    //     info!("cpu msm takes {:?}", start.elapsed());
    //     info!("cpu total takes {:?}", cpu_start.elapsed());

    //     let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[0]).unwrap();
    //     let gpu_poly = GpuPoly::<Fr>::new(&poly_kern, MAX_LENGTH).unwrap();

    //     let mut kern = MultiexpKernel::<G1Affine>::create(1).unwrap();

    //     // let window_size = 8;
    //     // crate::cuda::multiexp::generate_bases::<G1Affine, G1Affine>(
    //     //     &g,
    //     //     "g_prepared.dat",
    //     //     window_size,
    //     // );
    //     // crate::cuda::multiexp::generate_bases::<G1Affine, G1Affine>(
    //     //     &g,
    //     //     "shifted_g_prepared.dat",
    //     //     window_size,
    //     // );

    //     info!("setting up bases");
    //     kern.setup_base(GPU_OP::SETUP_G).unwrap();
    //     kern.setup_base(GPU_OP::SETUP_SHIFTED_G).unwrap();

    //     let start = Instant::now();
    //     let mut iter = 1;

    //     let end = MAX_LENGTH;

    //     loop {
    //         kern
    //             .multiexp_precalc(&g[..end], &v[..end], 0, end, GPU_OP::REUSE_G, true)
    //             .unwrap();
    //         iter += 1;
    //         if iter > 500 {
    //             break;
    //         }
    //     }

    //     let gpu = kern
    //         .multiexp_precalc(&g, &v, 0, MAX_LENGTH, GPU_OP::REUSE_G, true)
    //         .unwrap();

    //     info!(
    //         "gpu msm takes {:?}, average time = {}us",
    //         start.elapsed(),
    //         start.elapsed().as_micros() / iter
    //     );
    //     assert_eq!(cpu.into_affine(), gpu.into_affine());

    //     let start = Instant::now();
    //     let gpu = kern
    //         .multiexp_precalc(&g, &v1, 0, MAX_LENGTH, GPU_OP::REUSE_G, false)
    //         .unwrap();

    //     info!("gpu msm takes {:?}", start.elapsed());
    //     assert_eq!(cpu.into_affine(), gpu.into_affine());

    //     let start = Instant::now();
    //     let gpu = kern
    //         .multiexp_precalc(&g, &v, 0, MAX_LENGTH, GPU_OP::REUSE_SHIFTED_G, true)
    //         .unwrap();

    //     info!("gpu msm takes {:?}", start.elapsed());
    //     assert_eq!(cpu.into_affine(), gpu.into_affine());

    //     let start = Instant::now();
    //     let gpu = kern
    //         .multiexp_precalc(&g, &v1, 0, MAX_LENGTH, GPU_OP::REUSE_SHIFTED_G, false)
    //         .unwrap();

    //     info!("gpu msm takes {:?}", start.elapsed());
    //     assert_eq!(cpu.into_affine(), gpu.into_affine());

    //     let tmp_memory =
    //         unsafe { std::mem::transmute::<_, &DeviceMemory<GpuFr>>(gpu_poly.get_memory()) };
    //     gpu_poly.read_from(&v).unwrap();
    //     let start = Instant::now();

    //     let mut iter = 1;
    //     loop {
    //         kern.multiexp_precalc_gpu_ptr(
    //             &g,
    //             tmp_memory,
    //             0,
    //             MAX_LENGTH,
    //             GPU_OP::REUSE_SHIFTED_G,
    //             true,
    //         )
    //         .unwrap();

    //         iter += 1;
    //         if iter > 500 {
    //             break;
    //         }
    //     }

    //     info!(
    //         "gpu msm ptr mont takes {:?}, average time = {}us",
    //         start.elapsed(),
    //         start.elapsed().as_micros() / iter
    //     );

    //     assert_eq!(cpu.into_affine(), gpu.into_affine());

    //     gpu_poly.read_from(&v1).unwrap();
    //     let start = Instant::now();
    //     let gpu = kern
    //         .multiexp_precalc_gpu_ptr(
    //             &g,
    //             tmp_memory,
    //             0,
    //             MAX_LENGTH,
    //             GPU_OP::REUSE_SHIFTED_G,
    //             false,
    //         )
    //         .unwrap();

    //     info!("gpu msm precalc ptr not mont takes {:?}", start.elapsed());
    //     assert_eq!(cpu.into_affine(), gpu.into_affine());
    // }

    #[test]
    #[serial]
    pub fn test_gpu_precalc_ed() {
        let _ = env_logger::try_init();

        use std::time::Instant;

        let mut fr_rng = ark_std::test_rng();
        let mut rng = ark_std::test_rng();

        let precalc_container =
            MsmPrecalcContainer::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX]).unwrap();
        let mut kern = MultiexpKernel::<G1Affine>::create(GPU_DEV_IDX, &precalc_container).unwrap();
        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX]).unwrap();
        let mut gpu_poly = GpuPoly::<Fr>::new(&poly_kern, MAX_LENGTH).unwrap();

        info!("n = {}", MAX_LENGTH);
        info!("start generating fr");
        let v = (0..MAX_LENGTH).map(|_| fr_rng.gen()).collect::<Vec<Fr>>();

        info!("start generating affine points");
        let g = (0..MAX_LENGTH)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect::<Vec<G1Affine>>();

        let window_size = 8;

        // crate::cuda::multiexp::generate_ed_bases::<G1Affine>(
        //     &g,
        //     "ed_g_prepared.dat",
        //     window_size,
        // );

        // crate::cuda::multiexp::generate_ed_bases::<G1Affine>(
        //     &g,
        //     "ed_shifted_g_prepared.dat",
        //     window_size,
        // );

        // crate::cuda::multiexp::generate_ed_bases::<G1Affine>(
        //     &g,
        //     "ed_shifted_lagrange_g_prepared.dat",
        //     window_size,
        // );

        info!("setting up bases");
        let (acc_points_1, window_size) = kern.setup_ed_base(GPU_OP::SETUP_G).unwrap();
        let (acc_points_2, _) = kern.setup_ed_base(GPU_OP::SETUP_SHIFTED_G).unwrap();
        let (acc_points_3, _) = kern.setup_ed_base(GPU_OP::SETUP_SHIFTED_LAGRANGE_G).unwrap();

        kern.setup_ed_acc_points(GPU_OP::REUSE_G, &acc_points_1, window_size).unwrap();
        kern.setup_ed_acc_points(GPU_OP::REUSE_SHIFTED_G, &acc_points_2, window_size).unwrap();
        kern.setup_ed_acc_points(GPU_OP::REUSE_SHIFTED_LAGRANGE_G, &acc_points_3, window_size).unwrap();

        let cpu_start = Instant::now();
        let start = Instant::now();
        let v1 = v.iter().map(|v| v.into_repr()).collect::<Vec<_>>();

        info!("cpu into_repr takes {:?}", start.elapsed());

        let start = Instant::now();
        let cpu = VariableBaseMSM::multi_scalar_mul(&g, &v1);

        info!("cpu = {:?}", cpu);
        info!("cpu msm takes {:?}", start.elapsed());
        info!("cpu total takes {:?}", cpu_start.elapsed());

        let start = Instant::now();
        let iter = 1;

        let gpu = kern
            .multiexp_precalc_ed(&g, &v, 0, MAX_LENGTH, GPU_OP::REUSE_G, true)
            .unwrap();

        info!(
            "gpu msm ed takes {:?}, average time = {}us",
            start.elapsed(),
            start.elapsed().as_micros() / iter
        );
        assert_eq!(cpu.into_affine(), gpu.into_affine());

        let start = Instant::now();
        let gpu = kern
            .multiexp_precalc_ed(&g, &v1, 0, MAX_LENGTH, GPU_OP::REUSE_G, false)
            .unwrap();

        info!("gpu msm ed takes {:?}", start.elapsed());
        assert_eq!(cpu.into_affine(), gpu.into_affine());

        let start = Instant::now();
        let gpu = kern
            .multiexp_precalc_ed(&g, &v, 0, MAX_LENGTH, GPU_OP::REUSE_SHIFTED_G, true)
            .unwrap();

        info!("gpu msm ed takes {:?}", start.elapsed());
        assert_eq!(cpu.into_affine(), gpu.into_affine());

        let start = Instant::now();
        let gpu = kern
            .multiexp_precalc_ed(&g, &v1, 0, MAX_LENGTH, GPU_OP::REUSE_SHIFTED_G, false)
            .unwrap();

        info!("gpu msm ed takes {:?}", start.elapsed());
        assert_eq!(cpu.into_affine(), gpu.into_affine());

        let tmp_memory =
            unsafe { std::mem::transmute::<_, &DeviceMemory<GpuFr>>(gpu_poly.get_memory()) };
        gpu_poly.read_from(&v).unwrap();
        poly_kern.sync().unwrap();
        let start = Instant::now();

        let iter = 1;

        let gpu = kern
            .multiexp_precalc_ed_gpu_ptr(
                &g,
                tmp_memory,
                0,
                MAX_LENGTH,
                GPU_OP::REUSE_SHIFTED_G,
                true,
            )
            .unwrap();

        info!(
            "gpu msm ed ptr mont takes {:?}, average time = {}us",
            start.elapsed(),
            start.elapsed().as_micros() / iter
        );

        assert_eq!(cpu.into_affine(), gpu.into_affine());

        gpu_poly.read_from(&v1).unwrap();
        poly_kern.sync().unwrap();
        let start = Instant::now();
        let gpu = kern
            .multiexp_precalc_ed_gpu_ptr(
                &g,
                tmp_memory,
                0,
                MAX_LENGTH,
                GPU_OP::REUSE_SHIFTED_G,
                false,
            )
            .unwrap();

        info!(
            "gpu msm ed precalc ptr not mont takes {:?}",
            start.elapsed()
        );
        assert_eq!(cpu.into_affine(), gpu.into_affine());
    }
}
