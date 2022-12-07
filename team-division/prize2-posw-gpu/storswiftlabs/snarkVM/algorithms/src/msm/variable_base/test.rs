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


#[cfg(test)]
mod tests {
    use rand_xorshift::XorShiftRng;
    use snarkvm_curves::bls12_377::{Fr, G1Affine};
    use snarkvm_fields::{PrimeField};
    use snarkvm_utilities::rand::test_rng;

    use snarkvm_curves::{AffineCurve};
    use snarkvm_utilities::{test_crypto_rng};
    use crate::msm::variable_base::{batched, standard};
    use crate::msm::VariableBase;

    #[cfg(feature = "opencl")]
    use crate::msm::variable_base::{opencl};

    #[cfg(feature = "cuda")]
    use crate::msm::variable_base::{cuda};

    fn create_scalar_bases<G: AffineCurve<ScalarField=F>, F: PrimeField>(
        rng: &mut XorShiftRng,
        size: usize,
    ) -> (Vec<G>, Vec<F::BigInteger>) {
        let bases = (0..size).map(|_| G::rand(rng)).collect::<Vec<_>>();
        let scalars = (0..size).map(|_| F::rand(rng).to_repr()).collect::<Vec<_>>();
        (bases, scalars)
    }

    #[test]
    fn test_msm() {
        let mut rng = test_rng();
        let (bases, scalars) = create_scalar_bases::<G1Affine, Fr>(&mut rng, 1000);

        let naive_a = VariableBase::msm_naive(bases.as_slice(), scalars.as_slice());
        let naive_b = VariableBase::msm_naive_parallel(bases.as_slice(), scalars.as_slice());
        assert_eq!(naive_a, naive_b);

        let candidate = standard::msm(bases.as_slice(), scalars.as_slice());
        assert_eq!(naive_a, candidate);

        let candidate = batched::msm(bases.as_slice(), scalars.as_slice());
        assert_eq!(naive_a, candidate);
    }

    #[cfg(any(feature = "cuda", feature = "opencl"))]
    #[test]
    fn test_msm_gpu() {
        let basedata = [65537, 32767, 65537, 65535, 65535, 65535, 65537, 65537, 65537];
        let scalardata = [32764, 32767, 65534, 65535, 65535, 65535, 65535, 65535, 65533];

        for i in 0..9 {
            let base_size = basedata[i] as usize;
            let scalar_size = scalardata[i] as usize;
            //let data_size = 1<<16;
            //let (bases, scalars) = generate_points_scalars::<G1Affine>(data_size, 1);
            let (bases, scalars) = create_scalar_bases3::<G1Affine, Fr>(base_size, scalar_size);
            let coeffs = scalars.iter().map(|s| s.to_repr()).collect::<Vec<_>>();

            //eprintln!("test_msm_gpu   bases {} ", bases[0],);
            //eprintln!("test_msm_gpu scalars {} ", scalars[0].to_repr());
            eprintln!("test_msm_gpu bases scalars {}/{}", bases.len(), scalars.len());
            let rust = batched::msm(bases.as_slice(), coeffs.as_slice());

            let rust1 = standard::msm(bases.as_slice(), coeffs.as_slice());

            #[cfg(feature = "opencl")]
                let gpu = opencl::msm_opencl(bases.as_slice(), scalars.as_slice()).unwrap();

            #[cfg(feature = "cuda")]
                let gpu = cuda::msm_cuda(bases.as_slice(), scalars.as_slice()).unwrap();

            eprintln!("test_msm_gpu scpu {}", rust1);
            eprintln!("test_msm_gpu bcpu {}", rust);
            eprintln!("test_msm_gpu cuda {}", gpu);
            assert_eq!(rust, gpu);
        }
    }

    fn create_scalar_bases3<G: AffineCurve<ScalarField=F>, F: PrimeField>(
        len: usize,
        len2: usize,
    ) -> (Vec<G>, Vec<F>) {
        let mut rng = test_crypto_rng();

        let bases = (0..len).map(|_| G::rand(&mut rng)).collect::<Vec<_>>();
        let mut scalars = (0..len2).map(|_| F::rand(&mut rng)).collect::<Vec<_>>();
        if len2 <32000 {
            for i in 0..5000 {
                scalars[i+4123] = F::one();
            }
        }

        (bases, scalars)
    }


    #[test]
    fn test_msm_times() {
        use chrono::Local;
        use std::sync::atomic::{AtomicI64, Ordering};

        //let data_size = 1 << 16;
        let data = [26909, 26909, 32767, 32767, 65535, 65535, 65535, 65535, 65535, 65535, 65535];
        let mut dt_start = Local::now();
        let count = std::sync::Arc::new(AtomicI64::new(0));

        let mut n_thread = 6;
        if let Ok(t) = std::env::var("THREAD") {
            let m = t.parse::<i32>().unwrap();
            n_thread = if m >= 0 { m } else { 0 };
        }

        let mut h_thread = Vec::with_capacity(n_thread as usize);


        for i in 0..n_thread {
            let c1 = count.clone();
            h_thread.push(std::thread::spawn(move || {
                println!("start msm test sub thread {}", i + 1);

                let (bases1, scalars1) = create_scalar_bases3::<G1Affine, Fr>(26909, 26909);
                let (bases2, scalars2) = create_scalar_bases3::<G1Affine, Fr>(32767, 32767);
                let (bases3, scalars3) = create_scalar_bases3::<G1Affine, Fr>(65535, 65535);

                let coeffs1 = scalars1.iter().map(|s| s.to_repr()).collect::<Vec<_>>();
                let coeffs2 = scalars2.iter().map(|s| s.to_repr()).collect::<Vec<_>>();
                let coeffs3 = scalars3.iter().map(|s| s.to_repr()).collect::<Vec<_>>();


                let data_len = data.len();

                for idx in 0..(data_len * 400) {
                    let data_size = data[idx % data_len] as usize;
                    let mut bases = bases3.as_slice();
                    let mut scalars = scalars3.as_slice();
                    let mut coeffs = coeffs3.as_slice();
                    if data_size == 26909 {
                        bases = bases1.as_slice();
                        scalars = scalars1.as_slice();
                        coeffs = coeffs1.as_slice();
                    } else if data_size == 32767 {
                        bases = bases2.as_slice();
                        scalars = scalars2.as_slice();
                        coeffs = coeffs2.as_slice();
                    }

                    #[cfg(feature = "opencl")]
                    opencl::msm_opencl(bases, scalars).unwrap();

                    #[cfg(feature = "cuda")]
                    cuda::msm_cuda(bases, scalars).unwrap();

                    #[cfg(not(any(feature = "opencl", feature = "cuda")))]
                    batched::msm(bases, coeffs);

                    //assert_eq!(gpu, cpu);
                    if idx >= data_len * 3 && idx % data_len == 0 {
                        c1.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }));
        }

        println!("start msm test main thread");

        let (bases1, scalars1) = create_scalar_bases3::<G1Affine, Fr>(26909, 26909);
        let (bases2, scalars2) = create_scalar_bases3::<G1Affine, Fr>(32767, 32767);
        let (bases3, scalars3) = create_scalar_bases3::<G1Affine, Fr>(65535, 65535);

        let coeffs1 = scalars1.iter().map(|s| s.to_repr()).collect::<Vec<_>>();
        let coeffs2 = scalars2.iter().map(|s| s.to_repr()).collect::<Vec<_>>();
        let coeffs3 = scalars3.iter().map(|s| s.to_repr()).collect::<Vec<_>>();


        let data_len = data.len();
        for idx in 0..(data_len * 400) {
            let data_size = data[idx % data_len] as usize;
            let mut bases = bases3.as_slice();
            let mut scalars = scalars3.as_slice();
            let mut coeffs = coeffs3.as_slice();
            if data_size == 26909 {
                bases = bases1.as_slice();
                scalars = scalars1.as_slice();
                coeffs = coeffs1.as_slice();
            } else if data_size == 32767 {
                bases = bases2.as_slice();
                scalars = scalars2.as_slice();
                coeffs = coeffs2.as_slice();
            }

            if idx == data_len * 2 {
                dt_start = Local::now();
                count.store(0, Ordering::SeqCst);
            }

            #[cfg(feature = "opencl")]
            opencl::msm_opencl(bases, scalars).unwrap();

            #[cfg(feature = "cuda")]
            cuda::msm_cuda(bases, scalars).unwrap();

            #[cfg(not(any(feature = "opencl", feature = "cuda")))]
            batched::msm(bases, coeffs.as_slice());


            if idx >= data_len * 3 && idx % data_len == 0 {
                count.fetch_add(1, Ordering::SeqCst);
            }

            if idx % (5 * data_len) == 0 {
                let c3 = count.load(Ordering::SeqCst);
                let usetime = Local::now().timestamp_millis() - dt_start.timestamp_millis();
                eprintln!(" storswift {}  usetime: {}, per:{:.3} H/s", c3, usetime, (c3 as f64) * 1000f64 / (usetime as f64));
            }
        }

        for _ in 0..n_thread {
            if let Some(jh) = h_thread.pop() {
                jh.join().unwrap();
            }
        }
    }
}