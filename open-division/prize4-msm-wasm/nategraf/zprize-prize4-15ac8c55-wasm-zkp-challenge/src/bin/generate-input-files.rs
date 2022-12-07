use clap::Parser;
use std::path::Path;
use wasm_zkp_challenge::msm;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Output path to store the generated input vectors.
    #[clap(short, long, value_parser)]
    file: String,

    /// Count of input vectors to generate.
    #[clap(short, long, value_parser, default_value_t = 10)]
    count: usize,

    /// Number of elements, as a power of two, to include in each input vector.
    #[clap(short, long, value_parser, default_value_t = 12)]
    size: usize,
}

fn main() -> Result<(), msm::Error> {
    let args = Args::parse();

    let instances: Vec<_> = (0..args.count)
        .map(|_| msm::Instance::generate(1 << args.size))
        .collect();
    msm::write_instances(Path::new(&args.file), &instances, false)?;
    Ok(())
}
