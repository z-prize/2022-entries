use crate::structs::*;
use crate::error::*;

// copied from bellman-plonk
#[inline(always)]
pub fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

#[inline(always)]
pub fn bitreverse(n: usize, l: usize) -> usize {
    let mut r = n.reverse_bits();
    // now we need to only use the bits that originally were "last" l, so shift

    r >>= (std::mem::size_of::<usize>() * 8) - l;

    r
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Domain<F: PrimeField> {
    pub size: u64,
    pub power_of_two: u64,
    pub generator: F,
}

impl<F: PrimeField> Domain<F> {
    pub fn new_for_size(size: u64) -> GPUResult<Self> {
        let size = size.next_power_of_two();
        let mut power_of_two = 0;
        let mut k = size;
        while k != 1 {
            k >>= 1;
            power_of_two += 1;
        }

        let max_power_of_two = F::FftParams::TWO_ADICITY as u64;
        let mut generator = F::two_adic_root_of_unity();

        if power_of_two > max_power_of_two {
            return Err(GPUError::Simple("oversized domain"));
        }

        for _ in power_of_two..max_power_of_two {
            generator.square_in_place();
        }

        Ok(Self {
            size: size,
            power_of_two: power_of_two,
            generator: generator,
        })
    }
}

pub struct BitReversedOmegas<F: PrimeField> {
    pub omegas: Vec<F>,
    pub domain_size: usize,
}

impl<F: PrimeField> BitReversedOmegas<F> {
    pub fn new_for_domain(domain: &Domain<F>) -> Self {
        let domain_size = domain.size as usize;

        let omega = domain.generator;
        let precomputation_size = domain_size / 2;

        let log_n = log2_floor(precomputation_size);

        let mut omegas = vec![F::zero(); precomputation_size];
        let mut u = F::one();

        omegas.iter_mut().for_each(|v| {
            *v = u;
            u.mul_assign(&omega)
        });

        if omegas.len() > 2 {
            for k in 0..omegas.len() {
                let rk = bitreverse(k, log_n as usize);
                if k < rk {
                    omegas.swap(rk, k);
                }
            }
        }

        BitReversedOmegas {
            omegas,
            domain_size,
        }
    }
}

