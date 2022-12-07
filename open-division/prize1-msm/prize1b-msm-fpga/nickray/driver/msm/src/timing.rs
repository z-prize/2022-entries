use std::time::{Duration, SystemTime};

#[cfg(feature = "timings")]
#[inline]
pub fn timed<R>(name: &str, f: impl FnOnce() -> R) -> R {
    println!("{} ...", name);
    let t = SystemTime::now();
    let r = f();
    println!("... {:?}", t.elapsed());
    r
}

#[cfg(not(feature = "timings"))]
#[inline]
pub fn timed<R>(_: &str, f: impl FnOnce() -> R) -> R {
    f()
}

#[inline]
pub fn always_timed<R>(name: &str, f: impl FnOnce() -> R) -> R {
    println!("{} ...", name);
    let t = SystemTime::now();
    let r = f();
    println!("... {:?}", t.elapsed());
    r
}

#[inline]
pub fn elapsed<R>(name: &str, f: impl FnOnce() -> R) -> Duration {
    println!("{} ...", name);
    let t = SystemTime::now();
    let _ = f();
    let elapsed = t.elapsed().unwrap();
    println!("... {:?}", elapsed);
    elapsed
}
