#!/bin/bash

RUST_BACKTRACE=1 RUSTFLAGS=-Awarnings cargo test -- --nocapture
