fn main() {
    /* Link CUDA Driver API (libcuda.so) */

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64/stub");
    println!("cargo:rustc-link-lib=cuda");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
