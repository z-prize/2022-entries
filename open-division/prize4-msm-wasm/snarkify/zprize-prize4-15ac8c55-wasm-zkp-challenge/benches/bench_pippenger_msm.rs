use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::{Path, PathBuf};
use wasm_zkp_challenge::msm::{read_or_generate_instances, Instance};

mod perf;

const TEST_DIR_BASE: &'static str = "./.test";

fn bench_instance_path(count: usize, k: usize) -> PathBuf {
    Path::new(TEST_DIR_BASE)
        .join(format!("{}x{}", count, k))
        .join("instances")
}

const INPUT_SIZES: &'static [usize] = &[12];

fn bench_msm(c: &mut Criterion) {
    let functions: &[(&'static str, bool, &dyn Fn(&Instance))] = &[
        ("baseline", false, &|input: &Instance| {
            let _ = input.compute_msm_baseline();
        }),
        ("opt_false_false", false, &|input: &Instance| {
            let _ = input.compute_msm::<false, false>();
        }),
        ("opt_true_false", false, &|input: &Instance| {
            let _ = input.compute_msm::<true, false>();
        }),
        ("opt_true_true", true, &|input: &Instance| {
            let _ = input.compute_msm::<true, true>();
        }),
        ("opt_false_true", true, &|input: &Instance| {
            let _ = input.compute_msm::<false, true>();
        }),
    ];

    let mut group = c.benchmark_group("msm");
    for k in INPUT_SIZES.iter() {
        for (name, enabled, function) in functions {
            // Check to see if the bench is flagged as enabled above.
            if !enabled {
                continue;
            }

            let path = bench_instance_path(1, *k);
            let instances = read_or_generate_instances(&path, 1, 1 << k).unwrap();
            // I don't think black_box is needed based on what I am reading in the docs.
            // Shouldn't really hurt anything though, so I'll just leave it.
            let input = black_box(&instances[0]);

            group.throughput(Throughput::Elements(1 << k));
            group.bench_with_input(BenchmarkId::new(*name, k), &input, |b, input| {
                b.iter(|| {
                    let _res = function(input);
                })
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(perf::FlamegraphProfiler::new(100));
    targets = bench_msm
}
criterion_main!(benches);
