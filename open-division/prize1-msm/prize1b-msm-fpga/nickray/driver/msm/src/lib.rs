pub mod app;
pub use app::{App, Instruction};

pub mod digits;
pub use digits::{limb_carries, Digit, Limb, Scalar};

pub mod timing;
pub use timing::{always_timed, elapsed, timed};

pub mod twisted;

use our_bls12_377::{Fq, Fr, G1Affine, G1PTEAffine, G1Projective};

pub fn random_points(size: u8) -> Vec<G1Affine> {
    use rand_core::SeedableRng;
    let mut rng = rand::prelude::StdRng::from_entropy();

    use ark_std::UniformRand;
    let points: Vec<_> = timed("generating random projective points", || {
        (0..(1 << size))
            .map(|_| G1Projective::rand(&mut rng))
            .collect()
    });

    use our_ec::ProjectiveCurve;
    timed("batch converting to affine", || {
        G1Projective::batch_normalization_into_affine(&points)
    })
}

pub fn not_so_random_points(size: u8, actual: u8) -> Vec<G1Affine> {
    use rand_core::SeedableRng;
    let rng = rand::prelude::StdRng::from_entropy();

    timed("generating not-so-random projective points", || {
        let random_points = random_points(actual);

        use rand::distributions::Slice;
        let slice = Slice::new(&random_points).unwrap();
        use rand::Rng;
        rng.sample_iter(&slice).take(1 << size).copied().collect()
    })
}

pub fn not_so_random_preprocessed_points(size: u8, actual: u8) -> Vec<G1PTEAffine> {
    use rand_core::SeedableRng;
    let rng = rand::prelude::StdRng::from_entropy();

    use crate::twisted::into_preprocessed;
    timed("generating not-so-random projective points", || {
        let random_points: Vec<_> = random_points(actual)
            .iter()
            .map(into_preprocessed)
            .collect();

        use rand::distributions::Slice;
        let slice = Slice::new(&random_points).unwrap();
        use rand::Rng;
        rng.sample_iter(&slice).take(1 << size).copied().collect()
    })
}

/// Generates points of the form {P_i} = {\beta^i * g}, where g = basepoint
pub fn harness_points(size: usize) -> (Fr, Vec<G1PTEAffine>) {
    use rand_core::SeedableRng;
    let length = 1 << size;

    use ark_std::{One, UniformRand};
    let mut rng = rand::prelude::StdRng::from_entropy();

    let beta = Fr::rand(&mut rng);
    eprintln!("using beta: {}", beta);

    let scalars = timed("scalar gen", || {
        let mut scalars = Vec::with_capacity(length as _);
        scalars.push(Fr::one());
        scalars.push(beta);
        let mut last = beta;
        for _ in 2..length {
            last *= beta;
            scalars.push(last);
        }
        scalars
    });

    use our_ec::{msm::FixedBase, AffineCurve, ProjectiveCurve};

    let points = timed("point gen", || {
        let scalar_bits = 256;
        let g = G1Affine::prime_subgroup_generator();
        let window = FixedBase::get_mul_window_size(length);
        let table =
            FixedBase::get_window_table::<G1Projective>(scalar_bits, window, g.into_projective());
        FixedBase::msm::<G1Projective>(scalar_bits, window, &table, &scalars)
    });

    let points = timed("Projective -> Affine", || {
        G1Projective::batch_normalization_into_affine(&points)
    });

    // the slow part
    (
        beta,
        timed("Affine -> PTEAffine", || preprocess_points(&points)),
    )
}

pub fn preprocess_points(points: &[G1Affine]) -> Vec<G1PTEAffine> {
    let mut ppoints = vec![G1PTEAffine::junk(our_ff::Fp::from(0)); points.len()];

    const CHUNK: usize = 1 << 16;
    for (chunk_in, chunk_out) in points
        .chunks(CHUNK)
        .zip(ppoints.as_mut_slice().chunks_mut(CHUNK))
    {
        crate::twisted::into_preprocessed_batched(chunk_in, chunk_out);
    }
    ppoints
}

pub fn load_harness_points(size: usize, name: &str) -> (Fr, Vec<G1PTEAffine>) {
    let beta_name = format!("{}.beta", name);
    let points_name = format!("{}.points", name);

    let mut beta = Fr::default();
    load(&mut beta, &beta_name);

    let mut points = vec![G1PTEAffine::junk(Fq::from(0u16)); 1 << size];
    timed("loading", || load_slice(&mut points, &points_name));
    (beta, points)
}

pub fn digits_to_scalars(digits: &[Digit]) -> Vec<Fr> {
    digits
        .iter()
        .copied()
        .map(|digit| {
            if digit >= 0 {
                Fr::from(digit as u16)
            } else {
                -Fr::from((-(digit as i32)) as u16)
            }
        })
        .collect()
}

/// "Fast" calculation of MSM in SW via MSM of the scalars.
pub fn noconflict_harness_digits(beta: &Fr, size: usize) -> (Vec<i16>, G1Projective) {
    use ark_std::{One, Zero};

    use our_ec::AffineCurve;
    let g = G1Affine::prime_subgroup_generator();

    let digits = noconflict_column16(size as u8);
    let scalars = digits_to_scalars(&digits);

    // calculate expected result "in the exponent"
    let result = timed("SW MSM via betas", || {
        let mut beta_i = Fr::one();
        let mut prod = Fr::zero();
        for &scalar in &scalars {
            prod += scalar * beta_i;
            beta_i *= beta;
        }
        g.mul(prod)
    });

    (digits, result)
}

pub fn random_fr(size: u8) -> Vec<Fr> {
    use ark_std::UniformRand;
    use rand_core::SeedableRng;
    let mut rng = rand::prelude::StdRng::from_entropy();

    (0..(1 << size))
        .map(|_| {
            Fr::rand(&mut rng)
            // let scalar: &[u64] = &scalar.into_repr();
            // scalar.into_bigint().0
            // for limb in scalar.iter_mut() {
            //     *limb = rng.next_u64();
            // }
            // scalar
        })
        .collect()
}

pub fn harness_scalars(beta: &Fr, size: usize) -> (Vec<Scalar>, G1Projective) {
    use ark_std::{One, Zero};

    use our_ec::AffineCurve;
    use our_ff::PrimeField;
    let g = G1Affine::prime_subgroup_generator();

    let scalars = random_fr(size as u8);
    // let mut digits = random_digits(size as u8);
    // // digits[42] = i16::MIN;
    // // dbg!(digits[42]);
    // let scalars = digits_to_scalars(&digits);

    // calculate expected result "in the exponent"
    // \Sum_i (scalar_i * beta^i)
    let result = timed("SW MSM via betas", || {
        let mut beta_i = Fr::one();
        let mut prod = Fr::zero();
        for &scalar in &scalars {
            prod += scalar * beta_i;
            beta_i *= beta;
        }
        g.mul(prod)
    });

    let scalars: Vec<Scalar> = scalars
        .iter()
        .map(|scalar| scalar.into_bigint().0)
        .collect();

    (scalars, result)
}

pub fn harness_digits(beta: &Fr, size: usize) -> (Vec<i16>, G1Projective) {
    use ark_std::{One, Zero};

    use our_ec::AffineCurve;
    let g = G1Affine::prime_subgroup_generator();

    let digits = random_digits(size as u8);
    // digits[42] = i16::MIN;
    // dbg!(digits[42]);
    let scalars = digits_to_scalars(&digits);

    // calculate expected result "in the exponent"
    // \Sum_i (scalar_i * beta^i)
    let result = timed("SW MSM via betas", || {
        let mut beta_i = Fr::one();
        let mut prod = Fr::zero();
        for &scalar in &scalars {
            prod += scalar * beta_i;
            beta_i *= beta;
        }
        g.mul(prod)
    });

    (digits, result)
}

pub fn random_digits(size: u8) -> Vec<Digit> {
    use rand_core::{RngCore, SeedableRng};
    let mut rng = rand::prelude::StdRng::from_entropy();

    (0..(1 << size)).map(|_| rng.next_u32() as i16).collect()
}

pub fn noconflict_column16(size: u8) -> Vec<i16> {
    (0..(1usize << size)).map(|i| (i % 1024) as i16).collect()
}

pub fn random_scalars(size: u8) -> Vec<Scalar> {
    use rand_core::{RngCore, SeedableRng};
    let mut rng = rand::prelude::StdRng::from_entropy();

    (0..(1 << size))
        .map(|_| {
            let mut scalar = Scalar::default();
            for limb in scalar.iter_mut() {
                *limb = rng.next_u64();
            }
            scalar
        })
        .collect()
}

pub fn zero_scalars(size: u8) -> Vec<Scalar> {
    (0..(1 << size))
        .map(|_| {
            let mut scalar = Scalar::default();
            for limb in scalar.iter_mut() {
                *limb = 0;
            }
            scalar
        })
        .collect()
}

pub fn noconflict_scalars(size: u8) -> Vec<Scalar> {
    (0..(1 << size))
        .map(|i| {
            let mut scalar = Scalar::default();
            for limb in scalar.iter_mut() {
                let i = (i as u16 % 1024) as u64;
                *limb = (i << 48) | (i << 32) | (i << 16) | i
            }
            scalar
        })
        .collect()
}

pub fn store_slice<T: Sized>(slice: &[T], name: &str) {
    use std::io::Write as _;
    let slice_data_size = std::mem::size_of::<T>() * slice.len();
    std::fs::File::create(name)
        .unwrap()
        .write_all(unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice_data_size)
        })
        .unwrap();
    println!("wrote {} to {}", slice_data_size, name);
}

pub fn load_slice<T: Sized>(slice: &mut [T], name: &str) {
    use std::io::Read as _;
    let slice_data_size = std::mem::size_of::<T>() * slice.len();
    std::fs::File::open(name)
        .unwrap()
        .read_exact(unsafe {
            std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, slice_data_size)
        })
        .unwrap();
    println!("read {} from {}", slice_data_size, name);
}
pub fn store<T: Sized>(data: &T, name: &str) {
    use std::io::Write as _;
    let size = std::mem::size_of::<T>();
    std::fs::File::create(name)
        .unwrap()
        .write_all(unsafe { std::slice::from_raw_parts(data as *const T as *const u8, size) })
        .unwrap();
    println!("wrote {} to {}", size, name);
}

pub fn load<T: Sized>(data: &mut T, name: &str) {
    use std::io::Read as _;
    let size = std::mem::size_of::<T>();
    std::fs::File::open(name)
        .unwrap()
        .read_exact(unsafe { std::slice::from_raw_parts_mut(data as *mut T as *mut u8, size) })
        .unwrap();
    println!("read {} from {}", size, name);
}
