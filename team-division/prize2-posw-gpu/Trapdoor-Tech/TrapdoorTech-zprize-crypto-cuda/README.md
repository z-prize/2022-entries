# Usage

This repo aims to provide a convenient wrapper for calling Cuda APIs.

A proportion of the repo is based or inspired by [RustaCuda](https://github.com/bheisler/RustaCUDA)

## API Note

If you want to use a newer version of cuda header, follow these steps:

1. install `bindgen`

``` bash
$ cargo install bindgen
```

2. generate `cuda_bindgen.rs`

``` bash
$ bindgen /usr/local/cuda/targets/x86_64-linux/include/cuda.h -o cuda_bindgen.rs
```

3. replace the old `cuda_bindgen.rs` with the new one

``` bash
$ cp cuda_bindgen.rs src/api/
```
