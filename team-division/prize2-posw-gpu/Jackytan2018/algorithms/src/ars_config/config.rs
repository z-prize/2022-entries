extern crate serde_yaml;
use serde::{Deserialize, Serialize};
use rust_gpu_tools::Device; 

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct Config {
    pub max_test: usize,
    pub job_max: usize,
    pub num_rayon_cores_global: usize,
    pub msm_branch_limit: i32,
    pub msm_branch_max: i32,
    pub max_gpu_fft: usize,
    pub max_gpu_msm: usize,
    pub gpu_num: usize,
}

lazy_static::lazy_static! {
    pub static ref PROCESS_NUM: usize = 4;

    pub static ref CONFIG: Config = {
        Config::new_config()
    };
}

impl Config {
    pub fn new(
        max_test: usize,
        job_max: usize,
        num_rayon_cores_global: usize,
        msm_branch_limit: i32,
        msm_branch_max: i32,
        max_gpu_fft: usize,
        max_gpu_msm: usize
    ) -> Config{
        Config{
            max_test,
            job_max,
            num_rayon_cores_global,
            msm_branch_limit,
            msm_branch_max,
            max_gpu_fft,
            max_gpu_msm,
            gpu_num: Device::all().len(),
        }
    }

    pub fn new_config() -> Self {
        let config = Config::new(200, 5, 5, 10, 10, 1, 1);
        println!("config = {:?}", config);
        config
    }
}
