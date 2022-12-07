use ark_bls12_381::G1Affine;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use reference::msm::{compute_msm, generate_msm_inputs};

fn msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    for size in (8..20).step_by(2) {
        let (point_vec, scalar_vec) = generate_msm_inputs::<G1Affine>(1 << size);
        let point_vec = black_box(point_vec);
        let scalar_vec = black_box(scalar_vec);
        group.bench_function(format!("Input vector length: 2^{}", size), |b| {
            b.iter(|| {
                let _ = compute_msm::<G1Affine>(&point_vec, &scalar_vec);
            })
        });
    }
}

criterion_group!(benches, msm);
criterion_main!(benches);
