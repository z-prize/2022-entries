/*
 * Copyright (c) 2022 EulerSmile ( see https://github.com/EulerSmile).
 *
 * Dual-licensed under both the MIT and Apache-2.0 licenses;
 * you may not use this file except in compliance with the License.
 */

extern crate cc;

// Example custom build script.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_ARCH").unwrap() == "wasm32"
        && std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap() == "unknown"
        && std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "emscripten"
    {
        println!("cargo:rustc-link-arg=--no-entry");
        println!("cargo:rustc-link-arg=-s EXPORT_ES6=1");
        println!("cargo:rustc-link-arg=-s MODULARIZE=1");
        println!("cargo:rustc-link-arg=-o module.html");
        println!("cargo:rustc-env=EMCC_CFLAGS=-s ERROR_ON_UNDEFINED_SYMBOLS=0");
    }
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=core-wasm/ecp_BLS12381.c");
    cc::Build::new()
        .file("core-wasm/big_384_29.c")
        .file("core-wasm/fp_BLS12381.c")
        .file("core-wasm/ecp_BLS12381.c")
        .file("core-wasm/rom_curve_BLS12381.c")
        .file("core-wasm/rom_field_BLS12381.c")
        .file("core-wasm/msm_r.c")
        .opt_level(3)
        .flag_if_supported("-std=c99")
        .flag_if_supported("-Wall")
        .compile("mcore");
}
