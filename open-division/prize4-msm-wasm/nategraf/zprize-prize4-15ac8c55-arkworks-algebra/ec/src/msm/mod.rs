mod fixed_base;
mod variable_base;
mod variable_base_opt;
use ark_std::string::String;
pub use fixed_base::*;
pub use variable_base::*;
pub use variable_base_opt::*;

extern crate std;
use std::{
    env::var,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};
#[allow(missing_debug_implementations)]
pub struct MeasurementInfo {
    /// Show measurement
    pub show: bool,
    /// The start time
    pub time: Instant,
    /// What is being measured
    pub message: String,
    /// The indent
    pub indent: usize,
}

/// Global indent counter
pub static NUM_INDENT: AtomicUsize = AtomicUsize::new(0);

/// Gets the time difference between the current time and the passed in time
pub fn get_duration(start: Instant) -> usize {
    let final_time = Instant::now() - start;
    let secs = final_time.as_secs() as usize;
    let millis = final_time.subsec_millis() as usize;
    let micros = (final_time.subsec_micros() % 1000) as usize;
    secs * 1000000 + millis * 1000 + micros
}

/// Prints a measurement on screen
pub fn log_measurement(indent: Option<usize>, msg: String, duration: usize) {
    let indent = indent.unwrap_or(0);
    std::println!(
        "{}{} ........ {}s",
        "*".repeat(indent),
        msg,
        (duration as f32) / 1000000.0
    );
}

/// The result of this function is only approximately `ln(a)`
/// [`Explanation of usage`]
///
/// [`Explanation of usage`]: https://github.com/scipr-lab/zexe/issues/79#issue-556220473
fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}

/// Starts a measurement
pub fn start_measure(msg: String, always: bool) -> MeasurementInfo {
    let measure = env_value("MEASURE", 0);
    let indent = NUM_INDENT.fetch_add(1, Ordering::Relaxed);
    MeasurementInfo {
        show: always || measure == 1,
        time: Instant::now(),
        message: msg,
        indent,
    }
}

/// Stops a measurement, returns the duration
pub fn stop_measure(info: MeasurementInfo) -> usize {
    NUM_INDENT.fetch_sub(1, Ordering::Relaxed);
    let duration = get_duration(info.time);
    if info.show {
        log_measurement(Some(info.indent), info.message, duration);
    }
    duration
}

/// Gets the ENV variable if defined, otherwise returns the default value
pub fn env_value(key: &str, default: usize) -> usize {
    match var(key) {
        Ok(val) => val.parse().unwrap(),
        Err(_) => default,
    }
}
