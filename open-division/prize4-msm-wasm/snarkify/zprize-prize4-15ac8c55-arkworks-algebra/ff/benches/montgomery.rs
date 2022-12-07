use ark_ff::{
    fields::{Fp384, MontBackend, MontConfig},
    Field, UniformRand,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

// Use BLS12-381 base field as the benchmark target.
#[derive(MontConfig)]
#[modulus = "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787"]
#[generator = "2"]
pub struct FqConfig;
pub type Fq = Fp384<MontBackend<FqConfig, 6>>;

// Number of random values to use as inputs to the benchmark iterations.
// Criterion will determine how many iterations to run automatically. This setting simply increases
// the number of distinct inputs to avoid variance that might result for any particular single
// input.
const N: usize = 1000;

pub fn benchmark_square(c: &mut Criterion) {
    let rng = &mut ark_std::test_rng();

    let elements = (0..N).map(|_| Fq::rand(rng)).collect::<Vec<_>>();

    c.bench_with_input(BenchmarkId::new("square", N), &elements, |b, elements| {
        b.iter(|| {
            elements.into_iter().for_each(|x| {
                let _ = x.square();
            })
        });
    });
}

criterion_group!(benches, benchmark_square);
criterion_main!(benches);
