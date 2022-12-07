// use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use ark_bls12_377::{
    fr::Fr, G1Affine, G1PTEAffine, G1Projective as G1, G1TEAffine, G1TEProjective as G1TE,
};
use ark_ec::ProjectiveCurve as _; // into_affine
use ark_ff::fields::PrimeField as _; // into_bigint
use ark_std::UniformRand as _; // ::rand

use lab::msm::ema;

pub fn msm_benchmark(c: &mut Criterion) {
    let mut rng = ark_std::test_rng();

    let reg_point = G1::rand(&mut rng).into_affine();

    let ete_point = G1TE::rand(&mut rng).into_affine();

    let pete_point = G1PTEAffine::from(ete_point);

    let mut group = c.benchmark_group("ete");

    let ns = &[8usize]; //, 16, 18];
    for n in ns {
        let size = 1 << n;
        let reg_points: Vec<_> = (0..size).map(|_| reg_point).collect();
        let ete_points: Vec<_> = (0..size).map(|_| ete_point).collect();
        let pete_points: Vec<_> = (0..size).map(|_| pete_point).collect();

        // scalars are random though
        let scalars: Vec<_> = (0..size)
            .map(|_| Fr::rand(&mut rng).into_bigint())
            .collect();

        group
            // .measurement_time(Duration::from_secs(30))
            .sample_size(10)
            .bench_with_input(BenchmarkId::new("MSM-reg", n), &n, |b, _| {
                b.iter(|| {
                    let _ = <G1 as ark_ec::msm::VariableBaseMSM<G1Affine>>::msm_bigint(reg_points.as_slice(), &scalars);
                });
            })
            .bench_with_input(BenchmarkId::new("MSM-ete", n), &n, |b, _| {
                b.iter(|| {
                    let _ = <G1TE as ark_ec::msm::VariableBaseMSM<G1TEAffine>>::msm_bigint(ete_points.as_slice(), &scalars);
                });
            })
            .bench_with_input(BenchmarkId::new("MSM-pete", n), &n, |b, _| {
                b.iter(|| {
                    let _ = <G1TE as ark_ec::msm::VariableBaseMSM<G1PTEAffine>>::msm_bigint(pete_points.as_slice(), &scalars);
                });
            })
            .bench_with_input(BenchmarkId::new("MSM-ema", n), &n, |b, _| {
                b.iter(|| {
                    let _ = ema::msm_bigint_ema::<G1PTEAffine, G1TE>(pete_points.as_slice(), &scalars);
                });
            });
    }
}

criterion_group!(benches, msm_benchmark);
criterion_main!(benches);
