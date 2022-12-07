bitflags! {
    pub struct ContextFlags: u32 {
        const SCHED_AUTO = 0;
        const SCHED_SPIN = 1;
        const SCHED_YIELD = 2;
        const SCHED_BLOCKING_SYNC = 4;
        const SCHED_MASK = 7;
        const MAP_HOST = 8;
        const LMEM_RESIZE_TO_MAX = 16;
        const FLAGS_MASK = 31;
    }
}

bitflags! {
    pub struct StreamFlags: u32 {
        const DEFAULT = 0;
        const NON_BLOCKING = 1;
    }
}

bitflags! {
    pub struct HostMemoryFlags: u32 {
        const PORTABLE = 1;
        const DEVICEMAP = 2;
        const WRITECOMBINED = 4;
    }
}

/// common used featrues
pub enum DeviceAttribute {
    MaxThreadPerBlock = 1,
    MaxSharedMemoryPerBlock = 8,
    WarpSize = 10,
    MultiprocessorCount = 16,
    MaxThreadsPerMultiprocessor = 39,
    UnifiedAddressing = 41,
    MaxBlocksPerMultiprocessor = 106,
}
