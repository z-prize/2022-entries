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

#[macro_export]
macro_rules! cuda_error {
    () => {
        // Declare C/C++ counterpart as following:
        // extern "C" { fn foobar(...) -> cuda::Error; }
        mod cuda {
            #[repr(C)]
            pub struct Error {
                pub code: i32,
                str: Option<core::ptr::NonNull<i8>>, // just strdup("string") from C/C++
            }

            impl Drop for Error {
                fn drop(&mut self) {
                    extern "C" {
                        fn free(str: Option<core::ptr::NonNull<i8>>);
                    }
                    unsafe { free(self.str) };
                    self.str = None;
                }
            }

            impl From<Error> for String {
                fn from(status: Error) -> Self {
                    let c_str = if let Some(ptr) = status.str {
                        unsafe { std::ffi::CStr::from_ptr(ptr.as_ptr()) }
                    } else {
                        extern "C" {
                            fn cudaGetErrorString(code: i32) -> *const i8;
                        }
                        unsafe { std::ffi::CStr::from_ptr(cudaGetErrorString(status.code)) }
                    };
                    String::from(c_str.to_str().unwrap_or("unintelligible"))
                }
            }
        }
    };
}
