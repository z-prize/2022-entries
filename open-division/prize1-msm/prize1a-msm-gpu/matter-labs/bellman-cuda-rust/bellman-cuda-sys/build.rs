fn main() {
    cc::Build::new()
        .cuda(true)
        .define("THRUST_IGNORE_CUB_VERSION_CHECK", None)
        .include("native/cub")
        .flag("-arch=sm_86")
        .flag("--expt-relaxed-constexpr")
        .file("native/allocator.cu")
        .file("native/bellman-cuda-cub.cu")
        .file("native/bellman-cuda.cu")
        .file("native/ff_config.cu")
        .file("native/msm_kernels.cu")
        .file("native/msm.cu")
        .compile("bellman-cuda");
}
