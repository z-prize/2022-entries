//! # fpga
//!
//! Rust bindings to interact with AWS F1 FPGAs.

use core::ops;
#[cfg(feature = "f1")]
use core::ptr;

// use std::fs::File;

use thiserror::Error;

#[cfg(feature = "f1")]
pub use fpga_sys as sys;

pub mod align;
pub use align::{aligned, Aligned};

pub type SendBuffer = Aligned<[u8; 64]>;
pub type SendBuffer64 = Aligned<[u64; 8]>;

#[derive(Copy, Clone, Debug)]
pub struct ReceiveBuffer([u8; 56]);

impl ops::Deref for ReceiveBuffer {
    type Target = [u8; 56];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ops::DerefMut for ReceiveBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Default for ReceiveBuffer {
    fn default() -> Self {
        Self([0u8; 56])
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("FPGA drivers require running as root.")]
    SudoRequired,
}

pub type Result<T> = core::result::Result<T, Error>;

// const APP_TO_SIM: &str = "/home/centos/appxfer_app_to_sim";
// const SIM_TO_APP: &str = "/home/centos/appxfer_sim_to_app";

#[allow(unused_variables)]
pub trait Fpga {
    fn receive(&mut self, buffer: &mut ReceiveBuffer) {}
    fn receive_alloc(&mut self) -> ReceiveBuffer {
        let mut buffer = ReceiveBuffer::default();
        self.receive(&mut buffer);
        buffer
    }
    // pub fn raw_read(&self) -> [u32; 16];
    fn send(&mut self, index: usize, buffer: &SendBuffer) {}
    fn send64(&mut self, index: usize, buffer: &SendBuffer64) {}
    fn flush(&self) {}

    /// NB: still using offset, not index
    /// figure out if we need this
    fn write_register(&mut self, index: u32, x: u32) {}
    fn read_register(&self, index: u32) -> u32 {
        0
    }
}

#[derive(Copy, Clone)]
pub struct Null {
    __: (),
}

#[cfg(feature = "f1")]
#[derive(Copy, Clone)]
pub struct F1 {
    __: (),
    // cmdlog: Option<File>,
    // reglog: Option<File>,
    // recvlog: Option<File>,
}

#[derive(Copy, Clone)]
pub struct Sim {
    __: (),
}

impl Null {
    pub fn new() -> Self {
        Self { __: () }
    }
}

impl Default for Null {
    fn default() -> Self {
        Self::new()
    }
}

impl Fpga for Null {}

#[cfg(feature = "f1")]
impl F1 {
    pub fn new(slot: i32, offset: i32) -> Result<Self> {
        let rc = unsafe { sys::init_f1(slot, offset) };
        if rc == 0 {
            // Ok(Self { __: (), cmdlog: None, reglog: None, recvlog: None })
            Ok(Self { __: () })
        } else {
            Err(Error::SudoRequired)
        }
    }
    // pub fn with_cmdlog(self, path: &str) -> Self {
    //     Self { __: (), cmdlog: Some(File::create(path).unwrap()), reglog: self.reglog, recvlog: self.recvlog }
    // }
    //
    // pub fn with_reglog(self, path: &str) -> Self {
    //     Self { __: (), cmdlog: self.cmdlog, reglog: Some(File::create(path).unwrap()), recvlog: self.recvlog }
    // }
    //
    // pub fn with_recvlog(self, path: &str) -> Self {
    //     Self { __: (), cmdlog: self.cmdlog, reglog: self.reglog, recvlog: Some(File::create(path).unwrap()) }
    // }
    //
    // fn cmdlog(&mut self, offset: usize, payload: &[u8]) {
    //     use std::io::Write;
    //     if let Some(log) = &mut self.cmdlog {
    //         log.write(format!("{:016X},", offset).as_bytes()).unwrap();
    //         for byte in payload.iter() {
    //             log.write(format!("{:02X}", byte).as_bytes()).unwrap();
    //         }
    //         log.write(b"\n").unwrap();
    //     }
    // }
    //
    // fn reglog(&mut self, offset: u32, value: u32) {
    //     use std::io::Write;
    //     if let Some(log) = &mut self.reglog {
    //         log.write(format!("{:016X},{:016X}\n", offset, value).as_bytes()).unwrap();
    //     }
    // }
    //
    // fn recvlog(&mut self, buffer: &ReceiveBuffer) {
    //     use std::io::Write;
    //     if let Some(log) = &mut self.recvlog {
    //         log.write(format!("{}\n", hex::encode_upper(&**buffer)).as_bytes()).unwrap();
    //     }
    // }
}

#[cfg(feature = "f1")]
impl Fpga for F1 {
    fn receive(&mut self, buffer: &mut ReceiveBuffer) {
        // blocking
        let p = unsafe { sys::dma_wait_512() };
        // p points to 16 u32s, where the first and the 8th are sequence numbers.
        for i in 0..7 {
            buffer[i << 2..][..4].copy_from_slice(
                unsafe { ptr::read_volatile(p.add(i + 1)) }
                    .to_le_bytes()
                    .as_slice(),
            );
            buffer[(i + 7) << 2..][..4].copy_from_slice(
                unsafe { ptr::read_volatile(p.add(i + 9)) }
                    .to_le_bytes()
                    .as_slice(),
            );
        }
        // if self.recvlog.is_some() {
        //     self.recvlog(buffer);
        // }
    }

    fn read_register(&self, index: u32) -> u32 {
        let offset = index << 2;
        unsafe { sys::read_32_f1(offset) }
    }

    // write an aligned 64-byte chunk of data to an offset
    // since offset must be 64B-aligned too, use index parameter,
    // using offset = index << 6
    fn send(&mut self, index: usize, buffer: &SendBuffer) {
        let offset = index << 6;
        let slice: &[u8] = &**buffer;
        unsafe { sys::write_512_f1(offset as u64, &slice[0] as *const _ as _) };
        // if self.cmdlog.is_some() {
        //     self.cmdlog(offset, slice);
        // }
    }

    fn send64(&mut self, index: usize, buffer: &SendBuffer64) {
        let offset = index << 6;
        let slice: &[u64] = &**buffer;
        unsafe { sys::write_512_f1(offset as u64, &slice[0] as *const _ as _) };
        // if self.cmdlog.is_some() {
        //     let slice = unsafe { core::slice::from_raw_parts(
        //             &slice[0] as *const u64 as *const u8,
        //             slice.len() * 8
        //             )};
        //     self.cmdlog(offset, slice);
        // }
    }

    fn write_register(&mut self, index: u32, value: u32) {
        let offset = index << 2;
        unsafe { sys::write_32_f1(offset, value) };
        // if self.reglog.is_some() {
        //     self.reglog(offset, value);
        // }
    }

    fn flush(&self) {
        unsafe { sys::write_flush() };
    }
}

// impl Sim {
//     pub fn new() -> Self {
//         let app_to_sim = ffi::CString::new(APP_TO_SIM.as_bytes()).unwrap();
//         let sim_to_app = ffi::CString::new(SIM_TO_APP.as_bytes()).unwrap();
//         unsafe { sys::init(
//             app_to_sim.as_ptr() as *mut _,
//             sim_to_app.as_ptr() as *mut _,
//         ) };
//         Self { __: () }
//     }
//
//     // pub fn receive_extended(&self) -> ([u32; 14], u32, u32) {
//     //     let mut data = [0u32; 14];
//     //
//     //     // blocking
//     //     let p = unsafe { sys::dma_wait_512() };
//     //     // p points to 16 u32s, where the first and the 8th are sequence numbers.
//     //     for i in 0..7 {
//     //         data[i] = unsafe { ptr::read_volatile(p.add(i + 1)) };
//     //         data[i + 7] = unsafe { ptr::read_volatile(p.add(i + 9)) };
//     //     }
//     //     let seq_a = unsafe { ptr::read_volatile(p) };
//     //     let seq_b = unsafe { ptr::read_volatile(p.add(8)) };
//     //     (data, seq_a, seq_b)
//     // }
//
// }

// impl Fpga for Sim {
//
//     pub fn receive(&self) -> [u32; 14] {
//         let mut data = [0u32; 14];
//
//         // blocking
//         let p = unsafe { sys::dma_wait_512() };
//         // p points to 16 u32s, where the first and the 8th are sequence numbers.
//         for i in 0..7 {
//             data[i] = unsafe { ptr::read_volatile(p.add(i + 1)) };
//             data[i + 7] = unsafe { ptr::read_volatile(p.add(i + 9)) };
//         }
//         data
//     }
//
//     // write an aligned 64-byte chunk of data to an offset
//     // since offset must be 64B-aligned too, use index parameter,
//     // using offset = index << 6
//     pub fn send(&self, index: usize, buffer: &mut SendBuffer) {
//         // write_512 must be 64-byte aligned
//         let offset = index << 6;
//         unsafe { sys::write_512(offset as u32, &mut *buffer as *mut _ as _) };
//     }
//
// }
