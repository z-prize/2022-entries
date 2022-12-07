use bindgen::callbacks::{EnumVariantValue, ParseCallbacks};
use std::fs;
use std::path::PathBuf;

fn cuda_include_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "/include")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/include"
    }
}

fn cuda_lib_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        concat!(env!("CUDA_PATH"), "/lib/x64")
    }

    #[cfg(target_os = "linux")]
    {
        "/usr/local/cuda/lib64"
    }
}

#[derive(Debug)]
struct CudaParseCallbacks;

impl ParseCallbacks for CudaParseCallbacks {
    fn enum_variant_name(
        &self,
        _enum_name: Option<&str>,
        _original_variant_name: &str,
        _variant_value: EnumVariantValue,
    ) -> Option<String> {
        if let Some(enum_name) = _enum_name {
            match enum_name {
                "enum cudaDeviceAttr" => Some(_original_variant_name[11..].to_string()),
                "enum cudaError" => Some(_original_variant_name[4..].to_string()),
                "enum cudaMemcpyKind" => Some(_original_variant_name[10..].to_string()),
                "enum cudaMemPoolAttr" => Some(_original_variant_name[11..].to_string()),
                "enum cudaMemLocationType" => Some(_original_variant_name[19..].to_string()),
                "enum cudaMemAllocationType" => Some(_original_variant_name[21..].to_string()),
                "enum cudaMemAllocationHandleType" => {
                    Some(_original_variant_name[17..].to_string())
                }
                "enum cudaMemoryType" => Some(_original_variant_name[14..].to_string()),
                "enum cudaMemAccessFlags" => Some(_original_variant_name[22..].to_string()),
                _ => None,
            }
        } else {
            None
        }
    }

    fn item_name(&self, _original_item_name: &str) -> Option<String> {
        match _original_item_name {
            "cudaDeviceAttr" => Some("CudaDeviceAttr".to_string()),
            "cudaError" => Some("CudaError".to_string()),
            "cudaDeviceProp" => Some("CudaDeviceProperties".to_string()),
            "cudaMemcpyKind" => Some("CudaMemoryCopyKind".to_string()),
            "cudaMemPoolProps" => Some("CudaMemPoolProperties".to_string()),
            "cudaMemPoolAttr" => Some("CudaMemPoolAttribute".to_string()),
            "cudaMemLocation" => Some("CudaMemLocation".to_string()),
            "cudaMemLocationType" => Some("CudaMemLocationType".to_string()),
            "cudaMemAllocationType" => Some("CudaMemAllocationType".to_string()),
            "cudaMemAllocationHandleType" => Some("CudaMemAllocationHandleType".to_string()),
            "cudaPointerAttributes" => Some("CudaPointerAttributes".to_string()),
            "cudaMemoryType" => Some("CudaMemoryType".to_string()),
            "cudaMemAccessFlags" => Some("CudaMemAccessFlags".to_string()),
            "cudaMemAccessDesc" => Some("CudaMemAccessDesc".to_string()),
            _ => None,
        }
    }
}

fn main() {
    let cuda_runtime_api_path = PathBuf::from(cuda_include_path())
        .join("cuda_runtime_api.h")
        .to_string_lossy()
        .to_string();
    println!("cargo:rustc-link-search=native={}", cuda_lib_path());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed={}", cuda_runtime_api_path);

    let bindings = bindgen::Builder::default()
        .header(cuda_runtime_api_path)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .parse_callbacks(Box::new(CudaParseCallbacks))
        .size_t_is_usize(true)
        .generate_comments(false)
        .rustfmt_bindings(true)
        .layout_tests(false)
        .allowlist_type("cudaError")
        .rustified_enum("cudaError")
        .must_use_type("cudaError")
        // device management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
        .rustified_enum("cudaDeviceAttr")
        .allowlist_function("cudaDeviceGetAttribute")
        .allowlist_function("cudaDeviceGetDefaultMemPool")
        .allowlist_function("cudaDeviceGetMemPool")
        .allowlist_function("cudaDeviceReset")
        .allowlist_function("cudaDeviceSetMemPool")
        .allowlist_function("cudaDeviceSynchronize")
        .allowlist_function("cudaGetDevice")
        .allowlist_function("cudaGetDeviceCount")
        .allowlist_function("cudaGetDeviceProperties")
        .allowlist_function("cudaSetDevice")
        // error handling
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
        .allowlist_function("cudaGetErrorName")
        .allowlist_function("cudaGetLastError")
        .allowlist_function("cudaPeekAtLastError")
        // stream management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        .allowlist_function("cudaStreamCreate")
        .allowlist_var("cudaStreamDefault")
        .allowlist_var("cudaStreamNonBlocking")
        .allowlist_function("cudaStreamCreateWithFlags")
        .allowlist_function("cudaStreamDestroy")
        .allowlist_function("cudaStreamQuery")
        .allowlist_function("cudaStreamSynchronize")
        .allowlist_var("cudaEventWaitDefault")
        .allowlist_var("cudaEventWaitExternal")
        .allowlist_function("cudaStreamWaitEvent")
        // event management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
        .allowlist_function("cudaEventCreate")
        .allowlist_var("cudaEventDefault")
        .allowlist_var("cudaEventBlockingSync")
        .allowlist_var("cudaEventDisableTiming")
        .allowlist_var("cudaEventInterprocess")
        .allowlist_function("cudaEventCreateWithFlags")
        .allowlist_function("cudaEventDestroy")
        .allowlist_function("cudaEventElapsedTime")
        .allowlist_function("cudaEventQuery")
        .allowlist_function("cudaEventRecord")
        .allowlist_var("cudaEventRecordDefault")
        .allowlist_var("cudaEventRecordExternal")
        .allowlist_function("cudaEventRecordWithFlags")
        .allowlist_function("cudaEventSynchronize")
        // execution control
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
        .allowlist_function("cudaLaunchHostFunc")
        // memory management
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
        .rustified_enum("cudaMemcpyKind")
        .allowlist_function("cudaFree")
        .allowlist_function("cudaFreeHost")
        .allowlist_var("cudaHostAllocDefault")
        .allowlist_var("cudaHostAllocPortable")
        .allowlist_var("cudaHostAllocMapped")
        .allowlist_var("cudaHostAllocWriteCombined")
        .allowlist_function("cudaHostAlloc")
        .allowlist_var("cudaHostRegisterDefault")
        .allowlist_var("cudaHostRegisterPortable")
        .allowlist_var("cudaHostRegisterMapped")
        .allowlist_var("cudaHostRegisterIoMemory")
        .allowlist_var("cudaHostRegisterReadOnly")
        .allowlist_function("cudaHostRegister")
        .allowlist_function("cudaHostUnregister")
        .allowlist_function("cudaMalloc")
        .allowlist_function("cudaMemGetInfo")
        .allowlist_function("cudaMemcpy")
        .allowlist_function("cudaMemcpyAsync")
        .allowlist_function("cudaMemset")
        .allowlist_function("cudaMemsetAsync")
        // Stream Ordered Memory Allocator
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
        .allowlist_function("cudaFreeAsync")
        .allowlist_function("cudaMallocAsync")
        .allowlist_function("cudaMallocFromPoolAsync")
        .rustified_enum("cudaMemLocationType")
        .rustified_enum("cudaMemAllocationType")
        .rustified_enum("cudaMemAllocationHandleType")
        .allowlist_function("cudaMemPoolCreate")
        .allowlist_function("cudaMemPoolDestroy")
        .rustified_enum("cudaMemPoolAttr")
        .rustified_enum("cudaMemAccessFlags")
        .allowlist_function("cudaMemPoolGetAccess")
        .allowlist_function("cudaMemPoolGetAttribute")
        .allowlist_function("cudaMemPoolSetAccess")
        .allowlist_function("cudaMemPoolSetAttribute")
        .allowlist_function("cudaMemPoolTrimTo")
        // Unified Addressing
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html
        .rustified_enum("cudaMemoryType")
        .allowlist_function("cudaPointerGetAttributes")
        // Peer Device Memory Access
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html
        .allowlist_function("cudaDeviceCanAccessPeer")
        .allowlist_function("cudaDeviceDisablePeerAccess")
        .allowlist_function("cudaDeviceEnablePeerAccess")
        //
        .generate()
        .expect("Unable to generate bindings");

    fs::write(
        PathBuf::from("src").join("bindings.rs"),
        bindings.to_string(),
    )
    .expect("Couldn't write bindings!");
}
