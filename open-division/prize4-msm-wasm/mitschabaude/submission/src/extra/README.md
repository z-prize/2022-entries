# Extra stuff

This folder contains miscellaneous code which currently doesn't contribute to any core functionality of this library, or has some other reason to be tucked away.

This also includes code that is really useful for testing, debugging and benchmarking, and is in active use by some of the scripts used to develop this library:

- the Arkworks reference implementation, compiled to Wasm from Rust
- the `tictoc.js` micro-library which makes timing performance much more convenient than `let t = performance.now()`
- a "dumb" MSM implementation based on JS bigints, non-batched affine additions and plain (non-pippenger) scalar multiplications, which is simple to reason about
