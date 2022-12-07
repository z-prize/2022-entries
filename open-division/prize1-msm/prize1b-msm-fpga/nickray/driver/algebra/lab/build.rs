extern crate rustc_version;

use rustc_version::{version_meta, Channel};

fn main() {
    // println!("cargo:rustc-link-search=all=/home/ec2-user/ema/zprize-msm/algebra/
    // lab"); println!("cargo:rustc-link-lib=dylib=doubler.o");

    if version_meta().expect("nightly check failed").channel == Channel::Nightly {
        println!("cargo:rustc-cfg=nightly");
    }
}
