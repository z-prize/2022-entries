use ark_bls12_381::fq::FqConfig;
use ark_ff::MontConfig;

fn main() {
    println!(
        "FqConfig::CAN_USE_NO_CARRY_OPT: {}",
        FqConfig::CAN_USE_NO_CARRY_OPT
    );
    println!(
        "FqConfig::CAN_USE_SQUARE_NO_CARRY_OPT: {}",
        FqConfig::CAN_USE_SQUARE_NO_CARRY_OPT
    );
    #[cfg(feature = "partial-reduce")]
    println!(
        "FqConfig::CAN_USE_PARTIAL_REDUCE_OPT: {}",
        FqConfig::CAN_USE_PARTIAL_REDUCE_OPT
    );
    println!("FqConfig::MODULUS: {:X}", FqConfig::MODULUS);
    #[cfg(feature = "partial-reduce")]
    println!("FqConfig::REDUCTION_BOUND: {:X}", FqConfig::REDUCTION_BOUND);
}
