use clap::Parser;
use std::path::Path;
use wasm_zkp_challenge::msm;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input path where the input vector files can be found.
    #[clap(short, long, value_parser)]
    file: String,
}

fn main() -> Result<(), msm::Error> {
    let args = Args::parse();

    let instances = msm::read_instances(Path::new(&args.file))?;
    let deserialize_hash = msm::hash(&instances)?;
    println!("Hash of input files: {:?}", &deserialize_hash);
    Ok(())
}
