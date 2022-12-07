// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

#![allow(unused_imports)]

extern crate cc;

use std::env;
use std::path::{Path, PathBuf};

fn assembly(file_vec: &mut Vec<PathBuf>, base_dir: &Path, _: &String) {
    file_vec.push(base_dir.join("assembly.S"))
}

fn main() {
    // account for cross-compilation [by examining environment variable]
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    if target_arch.eq("wasm32") {
        println!("cargo:rustc-cfg=feature=\"no-threads\"");
    }

    /*
     * Use pre-built libblst.a if there is one. This is primarily
     * for trouble-shooting purposes. Idea is that libblst.a can be
     * compiled with flags independent from cargo defaults, e.g.
     * '../../build.sh -O1 ...'.
     */
    if Path::new("libblst.a").exists() {
        println!("cargo:rustc-link-search=.");
        println!("cargo:rustc-link-lib=blst");
        println!("cargo:rerun-if-changed=libblst.a");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = manifest_dir.parent().unwrap();
    let src_dir= manifest_dir.join("src");
    let mut out_path = src_dir.join("api");

    let blst_base_dir = manifest_dir.clone();

    // Set CC environment variable to choose alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.

    let mut cc = cc::Build::new();

    let c_src = out_path.join("exports.c");
    println!("cargo:rerun-if-changed={}", c_src.display());
    let mut file_vec = vec![c_src];

    if target_arch.eq("x86_64") || target_arch.eq("aarch64") {
        let asm_dir = blst_base_dir.join("assembly");
        println!("cargo:rerun-if-changed={}", asm_dir.display());
        assembly(&mut file_vec, &asm_dir, &target_arch);
    } else {
        cc.define("__BLST_NO_ASM__", None);
    }
    match (cfg!(feature = "portable"), cfg!(feature = "force-adx")) {
        (true, false) => {
            println!("Compiling in portable mode without ISA extensions");
            cc.define("__BLST_PORTABLE__", None);
        }
        (false, true) => {
            if target_arch.eq("x86_64") {
                println!("Enabling ADX support via `force-adx` feature");
                cc.define("__ADX__", None);
            } else {
                println!("`force-adx` is ignored for non-x86_64 targets");
            }
        }
        (false, false) => {
            #[cfg(target_arch = "x86_64")]
            if target_arch.eq("x86_64") && std::is_x86_feature_detected!("adx")
            {
                println!("Enabling ADX because it was detected on the host");
                cc.define("__ADX__", None);
            }
        }
        (true, true) => panic!(
            "Cannot compile with both `portable` and `force-adx` features"
        ),
    }
    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-Wno-unused-function")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if target_arch.eq("wasm32") {
        cc.flag_if_supported("-ffreestanding");
    }
    if !cfg!(debug_assertions) {
        cc.opt_level(2);
    }
    cc.files(&file_vec).out_dir(out_dir).compile("blst");

    // pass some DEP_BLST_* variables to dependents
    //println!(
    //    "cargo:BINDINGS={}",
    //    blst_base_dir.join("bindings").to_string_lossy()
    //);

    // Generate related bindings
    //let bindings = bindgen::Builder::default()
    //    .header(format!("{}/blst.h", out_path.to_string_lossy()))
    //    .parse_callbacks(Box::new(bindgen::CargoCallbacks))
    //    .generate()
    //    .expect("Unable to generate bindings");
 
    //bindings
    //    .write_to_file(out_path.join("blst_bindings.rs"))
    //    .expect("Couldn't write bindings!");
    println!("cargo:rustc-link-search={}", out_dir.to_string_lossy());
    println!("cargo:rustc-link-lib=static=blst");
    println!("cargo:rerun-if-changed={}/libblst.a", out_dir.to_string_lossy());
    println!("cargo:rerun-if-changed={}/blst.h", out_path.to_string_lossy());
    println!("cargo:rerun-if-changed={}/blst_bindings.rs", out_path.to_string_lossy());
}
