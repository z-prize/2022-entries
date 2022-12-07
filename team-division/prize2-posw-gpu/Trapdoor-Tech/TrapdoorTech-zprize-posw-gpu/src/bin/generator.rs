use ec_gpu_common::GPUError;
use snarkvm_algorithms::snark::marlin::{AHPForR1CS, MarlinNonHidingMode};
use snarkvm_curves::bls12_377::{Bls12_377Parameters, Fr};
use snarkvm_curves::templates::bls12::Bls12;
use snarkvm_curves::PairingEngine;
use snarkvm_dpc::testnet2::Testnet2;
use snarkvm_dpc::Network;
use snarkvm_dpc::PoSWScheme;

use ec_gpu_common::{Fr as GpuFr, GPUSourceCore, GpuPolyContainer, PolyKernel};
use snarkvm_algorithms::fft::EvaluationDomain;

use clap::Parser;

const MAX_WINDOW_SIZE: usize = 9;
const MIN_WINDOW_SIZE: usize = 4;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[clap(short, long)]
    window_size: usize,
    #[clap(short, long, action)]
    gen_shifted_lagrange_basis: bool,
}

fn load_shifted_lag_g<G: snarkvm_curves::AffineCurve>() -> Vec<G> {
    use snarkvm_utilities::io::BufReader;
    use snarkvm_utilities::{FromBytes, Read};
    use std::fs::File;
    use std::path::PathBuf;

    let mut file_reader = BufReader::new(
        File::open(PathBuf::from("lagrange_g_calced.data".to_owned()))
            .map_err(|_| GPUError::Simple("Cannot read lagrange_g_calced.data".to_owned()))
            .unwrap(),
    );

    let mut buffer = [0u8; 8];
    file_reader.read(&mut buffer[..]).unwrap();

    let mut shifted_lag_g = Vec::new();

    loop {
        let affine: std::io::Result<G> = FromBytes::read_le(&mut file_reader);

        if let Ok(affine) = affine {
            shifted_lag_g.push(affine);
        } else {
            break;
        }
    }

    shifted_lag_g
}

fn generate_files(window_len: usize) {
    let window_len = if window_len > MAX_WINDOW_SIZE {
        MAX_WINDOW_SIZE
    } else if window_len < MIN_WINDOW_SIZE {
        MIN_WINDOW_SIZE
    } else {
        window_len
    };

    println!("generating files for window_len = {}", window_len);

    // generating lookup tables
    let posw = Testnet2::posw();
    let pk = posw.proving_key().as_ref().unwrap().committer_key.clone();

    let basis = pk.lagrange_bases_at_beta_g.get(&(1 << 15)).unwrap();
    let shifted_basis =
        load_shifted_lag_g::<<<Testnet2 as Network>::InnerCurve as PairingEngine>::G1Affine>();

    println!("generating g precalculated table...");
    ec_gpu_common::generate_ed_bases(
        &pk.powers().powers_of_beta_g.as_ref(),
        "./ed_g_prepared.dat",
        window_len,
    );

    println!("generating shifted g precalculated table...");
    ec_gpu_common::generate_ed_bases(
        &pk.shifted_powers_of_beta_g.as_ref().unwrap(),
        "./ed_shifted_g_prepared.dat",
        window_len,
    );

    println!("generating lagrange g precalculated table...");
    ec_gpu_common::generate_ed_bases(basis.as_ref(), "./ed_lagrange_g_prepared.dat", window_len);

    println!("generating shifted lagrange g precalculated table...");
    ec_gpu_common::generate_ed_bases(
        shifted_basis.as_slice(),
        "./ed_shifted_lagrange_g_prepared.dat",
        window_len,
    );

    println!("all finished!");
}

// REMINDER: this function requires ~260G memory to speed up calculation
// REMINDER: make sure the memory requirement is met before running this function
fn generate_shifted_lagrange_basis() {
    // assume at least one GPU device can be used
    let core = GPUSourceCore::create_cuda(0).unwrap();

    let kern = PolyKernel::<GpuFr>::create_with_core(&core).unwrap();
    let mut container = GpuPolyContainer::<GpuFr>::create().unwrap();

    let posw = Testnet2::posw();
    let pk = &posw.proving_key().as_ref().unwrap().committer_key;

    // we need shifted g bases for commitment
    // thus we generate and fill up the shifted g precalc table
    println!("generating shifted g precalculated table...");
    let window_len = 8;
    ec_gpu_common::generate_ed_bases(
        pk.shifted_powers_of_beta_g.as_ref().unwrap(),
        "./ed_shifted_g_prepared.dat",
        window_len,
    );

    let n = pk.shifted_powers_of_beta_g.as_ref().unwrap().len();
    let domain = EvaluationDomain::<Fr>::new(n).expect("cannot instantialize such domain");
    let domain_size = domain.size();

    println!("using GPU to generate shifted_lagrange_g for n = {n}, domain_size = {domain_size}");

    AHPForR1CS::<Fr, MarlinNonHidingMode>::prepare_domain::<GpuFr>(
        &kern,
        &mut container,
        domain_size,
    )
    .unwrap();

    AHPForR1CS::<Fr, MarlinNonHidingMode>::prepare_lagrange_bases::<
        GpuFr,
        Bls12<Bls12_377Parameters>,
    >(
        &kern,
        &mut container,
        pk.shifted_powers_of_beta_g.as_ref().unwrap(),
        n,
        domain_size,
    )
    .unwrap();
}

fn main() {
    let args = Args::parse();
    let window_size = args.window_size;
    let gen_basis = args.gen_shifted_lagrange_basis;

    if gen_basis {
        generate_shifted_lagrange_basis()
    }

    generate_files(window_size);
}
