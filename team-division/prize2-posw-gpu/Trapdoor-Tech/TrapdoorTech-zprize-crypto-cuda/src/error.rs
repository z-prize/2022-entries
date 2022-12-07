use std::convert::From;
use std::error::Error;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_char;
use std::ptr;

use crate::api::CUresult;

pub type CudaResult<T> = Result<T, CudaError>;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum CudaError {
    CudaErrorInvalidValue = 1,
    CudaErrorOutOfMemory = 2,
    CudaErrorNotInitialized = 3,
    CudaErrorDeinitialized = 4,
    CudaErrorProfilerDisabled = 5,
    CudaErrorProfilerNotInitialized = 6,
    CudaErrorProfilerAlreadyStarted = 7,
    CudaErrorProfilerAlreadyStopped = 8,
    CudaErrorStubLibrary = 34,
    CudaErrorNoDevice = 100,
    CudaErrorInvalidDevice = 101,
    CudaErrorDeviceNotLicensed = 102,
    CudaErrorInvalidImage = 200,
    CudaErrorInvalidContext = 201,
    CudaErrorContextAlreadyCurrent = 202,
    CudaErrorMapFailed = 205,
    CudaErrorUnmapFailed = 206,
    CudaErrorArrayIsMapped = 207,
    CudaErrorAlreadyMapped = 208,
    CudaErrorNoBinaryForGpu = 209,
    CudaErrorAlreadyAcquired = 210,
    CudaErrorNotMapped = 211,
    CudaErrorNotMappedAsArray = 212,
    CudaErrorNotMappedAsPointer = 213,
    CudaErrorEccUncorrectable = 214,
    CudaErrorUnsupportedLimit = 215,
    CudaErrorContextAlreadyInUse = 216,
    CudaErrorPeerAccessUnsupported = 217,
    CudaErrorInvalidPtx = 218,
    CudaErrorInvalidGraphicsContext = 219,
    CudaErrorNvlinkUncorrectable = 220,
    CudaErrorJitCompilerNotFound = 221,
    CudaErrorUnsupportedPtxVersion = 222,
    CudaErrorJitCompilationDisabled = 223,
    CudaErrorUnsupportedExecAffinity = 224,
    CudaErrorInvalidSource = 300,
    CudaErrorFileNotFound = 301,
    CudaErrorSharedObjectSymbolNotFound = 302,
    CudaErrorSharedObjectInitFailed = 303,
    CudaErrorOperatingSystem = 304,
    CudaErrorInvalidHandle = 400,
    CudaErrorIllegalState = 401,
    CudaErrorNotFound = 500,
    CudaErrorNotReady = 600,
    CudaErrorIllegalAddress = 700,
    CudaErrorLaunchOutOfResources = 701,
    CudaErrorLaunchTimeout = 702,
    CudaErrorLaunchIncompatibleTexturing = 703,
    CudaErrorPeerAccessAlreadyEnabled = 704,
    CudaErrorPeerAccessNotEnabled = 705,
    CudaErrorPrimaryContextActive = 708,
    CudaErrorContextIsDestroyed = 709,
    CudaErrorAssert = 710,
    CudaErrorTooManyPeers = 711,
    CudaErrorHostMemoryAlreadyRegistered = 712,
    CudaErrorHostMemoryNotRegistered = 713,
    CudaErrorHardwareStackError = 714,
    CudaErrorIllegalInstruction = 715,
    CudaErrorMisalignedAddress = 716,
    CudaErrorInvalidAddressSpace = 717,
    CudaErrorInvalidPc = 718,
    CudaErrorLaunchFailed = 719,
    CudaErrorCooperativeLaunchTooLarge = 720,
    CudaErrorNotPermitted = 800,
    CudaErrorNotSupported = 801,
    CudaErrorSystemNotReady = 802,
    CudaErrorSystemDriverMismatch = 803,
    CudaErrorCompatNotSupportedOnDevice = 804,
    CudaErrorMpsConnectionFailed = 805,
    CudaErrorMpsRpcFailure = 806,
    CudaErrorMpsServerNotReady = 807,
    CudaErrorMpsMaxClientsReached = 808,
    CudaErrorMpsMaxConnectionsReached = 809,
    CudaErrorStreamCaptureUnsupported = 900,
    CudaErrorStreamCaptureInvalidated = 901,
    CudaErrorStreamCaptureMerge = 902,
    CudaErrorStreamCaptureUnmatched = 903,
    CudaErrorStreamCaptureUnjoined = 904,
    CudaErrorStreamCaptureIsolation = 905,
    CudaErrorStreamCaptureImplicit = 906,
    CudaErrorCapturedEvent = 907,
    CudaErrorStreamCaptureWrongThread = 908,
    CudaErrorTimeout = 909,
    CudaErrorGraphExecUpdateFailure = 910,
    CudaErrorUnknown = 999,
}

impl Error for CudaError {}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            // 0 => write!(f, "cuda execution succeeds"),
            other if (other as u32) <= 999 => {
                let value = other as u32;
                let mut ptr: *const c_char = ptr::null();
                unsafe {
                    crate::api::cuGetErrorString(
                        std::mem::transmute(value),
                        &mut ptr as *mut *const c_char,
                    );
                    let cstr = CStr::from_ptr(ptr);
                    write!(f, "{:?}: {:?}", self, cstr)
                }
            }

            _ => write!(f, "unknown error"),
        }
    }
}

impl From<CUresult> for CudaError {
    fn from(e: CUresult) -> Self {
        unsafe { std::mem::transmute(e) }
    }
}
