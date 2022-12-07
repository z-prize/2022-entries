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

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut base_dir = manifest_dir.join("sppark");
    if !base_dir.exists() {
        // Reach out to .., which is the root of the sppark repo.
        // Use an absolute path to avoid issues with relative paths
        // being treated as strings by `cc` and getting concatenated
        // in ways that reach out of the OUT_DIR.
        base_dir = manifest_dir
            .parent()
            .expect("can't access parent of current directory")
            .into();
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("ec").to_string_lossy()
        );
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("ff").to_string_lossy()
        );
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("msm").to_string_lossy()
        );
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("util").to_string_lossy()
        );
    }
    // pass DEP_SPPARK_* variables to dependents
    println!("cargo:ROOT={}", base_dir.to_string_lossy());
}
