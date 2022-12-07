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
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

fn get_git_version() -> String {
    let version = env::var("CARGO_PKG_VERSION").unwrap().to_string();

    let child = Command::new("git")
        .args(&["describe", "--always", "--dirty"])
        .output();
    match child {
        Ok(child) => {
            let buf = String::from_utf8(child.stdout).expect("failed to read stdout");
            return version + "-" + &buf;
        },
        Err(err) => {
            eprintln!("`git describe` err: {}", err);
            return version;
        }
    }
}

fn main() {
    let version = get_git_version();
    let mut f = File::create(
        Path::new(&env::var("OUT_DIR").unwrap())
            .join("VERSION")).unwrap();
    f.write_all(version.trim().as_bytes()).unwrap();
}
