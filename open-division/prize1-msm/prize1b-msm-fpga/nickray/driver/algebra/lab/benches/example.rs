use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bls12_377::{fr::Fr, G1Affine, G1Projective as G1};
use ark_ec::ProjectiveCurve as _; // into_affine
use ark_ff::fields::PrimeField as _; // into_bigint
use ark_std::UniformRand as _; // ::rand

use lab::msm::{ema, ray};

// pub fn mul_benchmark(c: &mut Criterion) {
// const N: usize = 20;
//
// let mut rng = ark_std::test_rng();
//
// let mut group = c.benchmark_group("msm");
//
// let mut a1_u64 = [
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// ];
// let b1_u64 = [
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// ];
// let mut a2_u64 = [
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// ];
// let b2_u64 = [
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// u64::rand(&mut rng),
// ];
//
// let mut a1_f64 = [
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// ];
// let b1_f64 = [
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// ];
// let mut a2_f64 = [
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// ];
// let b2_f64 = [
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// f64::rand(&mut rng),
// ];
//
// run this bench with `cargo bench "MSM"` (or more specific filter)
// group
// .sample_size(10)
// .bench_with_input(BenchmarkId::new("mul-2-u64", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_2_u64(black_box(&mut a1_u64), black_box(&b1_u64));
// let _ = ema::mul_assign_2_u64(black_box(&mut a2_u64), black_box(&b2_u64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-3-u64", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_3_u64(black_box(&mut a1_u64), black_box(&b1_u64));
// let _ = ema::mul_assign_3_u64(black_box(&mut a2_u64), black_box(&b2_u64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-4-u64", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_4_u64(black_box(&mut a1_u64), black_box(&b1_u64));
// let _ = ema::mul_assign_4_u64(black_box(&mut a2_u64), black_box(&b2_u64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-2-f64", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_2_f64(black_box(&mut a1_f64), black_box(&b1_f64));
// let _ = ema::mul_assign_2_f64(black_box(&mut a2_f64), black_box(&b2_f64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-3-f64", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_3_f64(black_box(&mut a1_f64), black_box(&b1_f64));
// let _ = ema::mul_assign_3_f64(black_box(&mut a2_f64), black_box(&b2_f64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-4-f64", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_4_f64(black_box(&mut a1_f64), black_box(&b1_f64));
// let _ = ema::mul_assign_4_f64(black_box(&mut a2_f64), black_box(&b2_f64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-2", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_2_u64(black_box(&mut a1_u64), black_box(&b1_u64));
// let _ = ema::mul_assign_2_f64(black_box(&mut a2_f64), black_box(&b2_f64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-3", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_3_f64(black_box(&mut a1_f64), black_box(&b1_f64));
// let _ = ema::mul_assign_3_u64(black_box(&mut a2_u64), black_box(&b2_u64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-4", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_4_u64(black_box(&mut a1_u64), black_box(&b1_u64));
// let _ = ema::mul_assign_4_f64(black_box(&mut a2_f64), black_box(&b2_f64));
// });
// })
// .bench_with_input(BenchmarkId::new("mul-3+4", N), &N, |b, _| {
// b.iter(|| {
// let _ = ema::mul_assign_3_u64(black_box(&mut a1_u64), black_box(&b1_u64));
// let _ = ema::mul_assign_4_f64(black_box(&mut a2_f64), black_box(&b2_f64));
// });
// });
// }

pub fn msm_benchmark(c: &mut Criterion) {
    const N: usize = 20;
    const POINTS: usize = 1 << N;

    let mut rng = ark_std::test_rng();

    let point = G1::rand(&mut rng).into_affine();

    // just uses same point
    let points: Vec<_> = (0..POINTS).map(|_| point).collect();

    // scalars are random though
    let scalars: Vec<_> = (0..POINTS)
        .map(|_| Fr::rand(&mut rng).into_bigint())
        .collect();

    let mut group = c.benchmark_group("msm");

    // run this bench with `cargo bench "MSM"` (or more specific filter)
    group
        .sample_size(10)
        // .bench_with_input(BenchmarkId::new("MSM-naive", N), &N, |b, _| {
        //     b.iter(|| {
        //         use ark_bls12_377::G1Affine;
        //         use ark_ec::AffineCurve as _;
        //         let _: G1Affine = points.iter().zip(scalars.iter()).map(|(point, scalar)| point.mul(*scalar).into_affine()).sum();
        //     });
        // })
        // .bench_with_input(BenchmarkId::new("MSM-orig", N), &N, |b, _| {
        //     b.iter(|| {
        //         let _ = <G1 as ark_ec::msm::VariableBaseMSM<G1Affine>>::msm_bigint(points.as_slice(), &scalars);
        //     });
        // })
        // .bench_with_input(BenchmarkId::new("MSM-ray", N), &N, |b, _| {
        //     b.iter(|| {
        //         let _ = ray::msm_bigint::<G1Affine, G1>(16, points.as_slice(), &scalars);
        //     });
        // })
        .bench_with_input(BenchmarkId::new("MSM-ema", N), &N, |b, _| {
            b.iter(|| {
                let _ = ema::msm_bigint_ema::<G1Affine, G1>(points.as_slice(), &scalars);
            });
        });
}

// criterion_group!(benches, mul_benchmark);
criterion_group!(benches, msm_benchmark);
criterion_main!(benches);
