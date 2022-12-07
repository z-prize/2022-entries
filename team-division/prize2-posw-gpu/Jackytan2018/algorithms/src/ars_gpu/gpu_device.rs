use hashbrown::HashSet;
use rust_gpu_tools::{Device, GPUError, Program};
use std::sync::Mutex;
use crate::ars_config::*; 
use msm_sppark::*;
use snarkvm_curves::AffineCurve;

pub struct ArsProgram {
    pub id: usize,
    pub p: Program,
}

pub struct GpuDevice {
    pub id: usize,
    pub fft_in_use: usize,
    pub ffts: Vec<ArsProgram>,
    pub msms: Vec<ArsProgram>,
    pub msm_sppark_15_1_in_use: usize,
    pub msm_sppark_16_1_in_use: usize,
    pub msm_sppark_15_1: Vec<MultiScalarMultContext>,
    pub msm_sppark_16_1: Vec<MultiScalarMultContext>,

    pub msm_sppark_15_2_in_use: usize,
    pub msm_sppark_16_2_in_use: usize,
    pub msm_sppark_15_2: Vec<MultiScalarMultContext>,
    pub msm_sppark_16_2: Vec<MultiScalarMultContext>,
}

pub struct DevMgr {
    pub cur_dev_idx: usize,
    pub fft_in_use: usize,    
    pub devs: Vec<GpuDevice>,
    #[cfg(feature = "gpu_time")]
    pub start_time: Instant,
}

lazy_static::lazy_static! {
    pub static ref GPU_MGR: Mutex<DevMgr> = Mutex::new(create_mgr());
}

// initialize global DevMgr
fn create_mgr() -> DevMgr {
    let mut mgr = DevMgr {
        cur_dev_idx: 0,
        fft_in_use: 0,        
        devs: vec![],
        #[cfg(feature = "gpu_time")]
        start_time: Instant::now(),
    };
    for i in 0..CONFIG.gpu_num {
        mgr.devs.push(GpuDevice { 
            id: i, 
            fft_in_use: 0,
            ffts: vec![], 
            msms: vec![],
            msm_sppark_15_1_in_use: 0,
            msm_sppark_16_1_in_use: 0,
            msm_sppark_15_1: vec![],
            msm_sppark_16_1: vec![],

            msm_sppark_15_2_in_use: 0,
            msm_sppark_16_2_in_use: 0,
            msm_sppark_15_2: vec![],
            msm_sppark_16_2: vec![]
        });
    }
    mgr
}

pub fn ars_fetch_fft_program(ars_load_cuda_program: fn(usize) -> Result<Program, GPUError> ) -> Result<ArsProgram, GPUError>{
    loop {
        let (program , flag) = GPU_MGR.lock().unwrap().ars_fetch_fft_program(ars_load_cuda_program);
        if flag {
            return program;
        }
    }
}

#[inline]
pub fn ars_recycle_fft_program(program: ArsProgram) {
    GPU_MGR.lock().unwrap().ars_recycle_fft_program(program);
}

pub fn ars_fetch_msm_context<G: AffineCurve>(points: &[G], bit_len: usize, msm_key: usize) -> Result<MultiScalarMultContext, GPUError> {
    loop {
        let (context , flag) = GPU_MGR.lock().unwrap().ars_fetch_msm_context(points, bit_len, msm_key);
        if flag {
            return context;
        }
    }
}

#[inline]
pub fn ars_recycle_msm_context(context: MultiScalarMultContext, bit_len: usize, msm_key: usize) {
    GPU_MGR.lock().unwrap().ars_recycle_msm_context(context, bit_len, msm_key);
}

impl DevMgr {
    pub fn ars_get_devs_len(&self) -> usize {
        self.devs.len()
    }

    #[inline]
    pub fn ars_fetch_fft_program(&mut self, ars_load_cuda_program :fn (usize) -> Result<Program, GPUError> ) -> (Result<ArsProgram, GPUError>, bool) {
        if self.devs.is_empty() {
            return (Err(GPUError::DeviceNotFound), true);
        }

        if self.devs[self.cur_dev_idx].fft_in_use >= CONFIG.max_gpu_fft {
            self.cur_dev_idx += 1;
            if self.cur_dev_idx >= self.devs.len() {
                self.cur_dev_idx = 0;
            }
            return (Err(GPUError::DeviceNotFound), false);
        }

        self.devs[self.cur_dev_idx].fft_in_use += 1;

        let e = self.devs[self.cur_dev_idx].ffts.pop();
        let id = self.cur_dev_idx;
        self.cur_dev_idx += 1;
        if self.cur_dev_idx >= self.devs.len() {
            self.cur_dev_idx = 0;
        }   

        #[cfg(feature = "gpu_time")]
        self.ars_rec_start_time();

        self.fft_in_use += 1;
        if e.is_none() {
            let p = ars_load_cuda_program(id );
            match p {
                Ok(program) => {
                    return (Ok(ArsProgram { id: id , p: program}), true);
                }
                Err(err) => {
					self.devs[id].fft_in_use -= 1;
					self.fft_in_use -= 1;
                    eprintln!("Error loading cuda program: {:?}", err);
                    return (Err(err), true);
                }
            }
        } else {
            return (Ok(e.unwrap()), true);
        }
    }

    #[inline]
    pub fn ars_recycle_fft_program(&mut self, program: ArsProgram) {
        self.fft_in_use -= 1;
        self.devs[program.id].fft_in_use -= 1;

        #[cfg(feature = "gpu_time")]
        self.ars_rec_end_time();

        self.devs[program.id].ffts.push(program);
    }    

    #[inline]
    pub fn ars_fetch_msm_context<G: AffineCurve>(&mut self, points: &[G], bit_len: usize, msm_key: usize) -> (Result<MultiScalarMultContext, GPUError>, bool) {
        if self.devs.is_empty() {
            return (Err(GPUError::DeviceNotFound), true);
        }

        if bit_len == 15 {
            if msm_key == 1 {
                if self.devs[self.cur_dev_idx].msm_sppark_15_1_in_use >= CONFIG.max_gpu_msm {
                    self.cur_dev_idx += 1;
                    if self.cur_dev_idx >= self.devs.len() {
                        self.cur_dev_idx = 0;
                    }
                    return (Err(GPUError::DeviceNotFound), false);
                }
                self.devs[self.cur_dev_idx].msm_sppark_15_1_in_use += 1;

                let id = self.cur_dev_idx;
                self.cur_dev_idx += 1;
                if self.cur_dev_idx >= self.devs.len() {
                    self.cur_dev_idx = 0;
                }

                #[cfg(feature = "gpu_time")]
                self.ars_rec_start_time();

                let len = self.devs[id].msm_sppark_15_1.len();
                if len > 0 {
                    for i in 0..len{
                        if self.devs[id].msm_sppark_15_1[i].is_used == false {
                            self.devs[id].msm_sppark_15_1[i].is_used = true;
                            self.devs[id].msm_sppark_15_1[i].obj_id = i;
                            return (Ok(self.devs[id].msm_sppark_15_1[i]), true);
                        }
                    }
                    let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = len;
                    self.devs[id].msm_sppark_15_1.push(context);
                    return (Ok(self.devs[id].msm_sppark_15_1[len]), true);
                }else{
                    let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = 0;
                    self.devs[id].msm_sppark_15_1.push(context);
                    return (Ok(self.devs[id].msm_sppark_15_1[0]), true);
                }
            }else{
                if self.devs[self.cur_dev_idx].msm_sppark_15_2_in_use >= CONFIG.max_gpu_msm {
                    self.cur_dev_idx += 1;
                    if self.cur_dev_idx >= self.devs.len() {
                        self.cur_dev_idx = 0;
                    }
                    return (Err(GPUError::DeviceNotFound), false);
                }
                self.devs[self.cur_dev_idx].msm_sppark_15_2_in_use += 1;

                let id = self.cur_dev_idx;
                self.cur_dev_idx += 1;
                if self.cur_dev_idx >= self.devs.len() {
                    self.cur_dev_idx = 0;
                }

                #[cfg(feature = "gpu_time")]
                self.ars_rec_start_time();

                let len = self.devs[id].msm_sppark_15_2.len();
                if len > 0 {
                    for i in 0..len{
                        if self.devs[id].msm_sppark_15_2[i].is_used == false {
                            self.devs[id].msm_sppark_15_2[i].is_used = true;
                            self.devs[id].msm_sppark_15_2[i].obj_id = i;
                            return (Ok(self.devs[id].msm_sppark_15_2[i]), true);
                        }
                    }
                    let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = len;
                    self.devs[id].msm_sppark_15_2.push(context);
                    return (Ok(self.devs[id].msm_sppark_15_2[len]), true);
                }else{
                    let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = 0;
                    self.devs[id].msm_sppark_15_2.push(context);
                    return (Ok(self.devs[id].msm_sppark_15_2[0]), true);
                }
            }
        }else{
            if msm_key == 1 {
                if self.devs[self.cur_dev_idx].msm_sppark_16_1_in_use >= CONFIG.max_gpu_msm {
                    self.cur_dev_idx += 1;
                    if self.cur_dev_idx >= self.devs.len() {
                        self.cur_dev_idx = 0;
                    }
                    return (Err(GPUError::DeviceNotFound), false);
                }
                self.devs[self.cur_dev_idx].msm_sppark_16_1_in_use += 1;

                let id = self.cur_dev_idx;
                self.cur_dev_idx += 1;
                if self.cur_dev_idx >= self.devs.len() {
                    self.cur_dev_idx = 0;
                }

                #[cfg(feature = "gpu_time")]
                self.ars_rec_start_time();

                let len = self.devs[id].msm_sppark_16_1.len();
                if len > 0 {
                    for i in 0..len{
                        if self.devs[id].msm_sppark_16_1[i].is_used == false {
                            self.devs[id].msm_sppark_16_1[i].is_used = true;
                            self.devs[id].msm_sppark_16_1[i].obj_id = i;
                            return (Ok(self.devs[id].msm_sppark_16_1[i]), true);
                        }
                    }
                    let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = len;
                    self.devs[id].msm_sppark_16_1.push(context);
                    return (Ok(self.devs[id].msm_sppark_16_1[len]), true);
                }else{
                    let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = 0;
                    self.devs[id].msm_sppark_16_1.push(context);
                    return (Ok(self.devs[id].msm_sppark_16_1[0]), true);
                }
            }else{
                if self.devs[self.cur_dev_idx].msm_sppark_16_2_in_use >= CONFIG.max_gpu_msm {
                    self.cur_dev_idx += 1;
                    if self.cur_dev_idx >= self.devs.len() {
                        self.cur_dev_idx = 0;
                    }
                    return (Err(GPUError::DeviceNotFound), false);
                }
                self.devs[self.cur_dev_idx].msm_sppark_16_2_in_use += 1;

                let id = self.cur_dev_idx;
                self.cur_dev_idx += 1;
                if self.cur_dev_idx >= self.devs.len() {
                    self.cur_dev_idx = 0;
                }

                #[cfg(feature = "gpu_time")]
                self.ars_rec_start_time();

                let len = self.devs[id].msm_sppark_16_2.len();
                if len > 0 {
                    for i in 0..len{
                        if self.devs[id].msm_sppark_16_2[i].is_used == false {
                            self.devs[id].msm_sppark_16_2[i].is_used = true;
                            self.devs[id].msm_sppark_16_2[i].obj_id = i;
                            return (Ok(self.devs[id].msm_sppark_16_2[i]), true);
                        }
                    }
                   let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = len;
                    self.devs[id].msm_sppark_16_2.push(context);
                    return (Ok(self.devs[id].msm_sppark_16_2[len]), true);
                }else{
                    let mut context = ars_multi_scalar_mult_init(id, points);
                    context.is_used = true;
                    context.bit_len = bit_len;
                    context.obj_id = 0;
                    self.devs[id].msm_sppark_16_2.push(context);
                    return (Ok(self.devs[id].msm_sppark_16_2[0]), true);
                }
            }
        }
    }

    #[inline]
    pub fn ars_recycle_msm_context(&mut self, context: MultiScalarMultContext, bit_len: usize, msm_key: usize) {
        if bit_len == 15 {
            if msm_key == 1 {
                if self.devs[context.gpu_id].msm_sppark_15_1_in_use > 0 {
                    self.devs[context.gpu_id].msm_sppark_15_1_in_use -= 1;
                }
                self.devs[context.gpu_id].msm_sppark_15_1[context.obj_id].is_used = false;
            }else{
                if self.devs[context.gpu_id].msm_sppark_15_2_in_use > 0 {
                    self.devs[context.gpu_id].msm_sppark_15_2_in_use -= 1;
                }
                self.devs[context.gpu_id].msm_sppark_15_2[context.obj_id].is_used = false;
            }
        }else{
            if msm_key == 1 {
                if self.devs[context.gpu_id].msm_sppark_16_1_in_use > 0 {
                    self.devs[context.gpu_id].msm_sppark_16_1_in_use -= 1;
                }
                self.devs[context.gpu_id].msm_sppark_16_1[context.obj_id].is_used = false;
            }else{
                if self.devs[context.gpu_id].msm_sppark_16_2_in_use > 0 {
                    self.devs[context.gpu_id].msm_sppark_16_2_in_use -= 1;
                }
                self.devs[context.gpu_id].msm_sppark_16_2[context.obj_id].is_used = false;
            }
        }

        #[cfg(feature = "gpu_time")]
        self.ars_rec_end_time();
    }


    #[cfg(feature = "gpu_time")]
    pub fn ars_rec_start_time(&mut self) {
        if self.fft_in_use == 0 {
            self.start_time = Instant::now();
        }
    }

    #[cfg(feature = "gpu_time")]
    pub fn ars_rec_end_time(&mut self) {
        if self.fft_in_use == 0 {
            println!("use GPU time = {:?}", self.start_time.elapsed());
        }
    }
}

pub fn ars_get_devs_len() -> usize {
    GPU_MGR.lock().unwrap().ars_get_devs_len()
}

pub fn ars_gpu_model_name() -> String {
    Device::all()[0].name()
}

pub struct MsgMgr {  
    pub receivers: HashSet<fn(String)> 
}

impl MsgMgr {
    pub fn new() -> Self {
        Self {receivers: HashSet::new()}
    }

    pub fn register_receiver(&mut self, receiver: fn(String)) { 
        self.receivers.insert(receiver);
    }

    pub fn unregister_receiver(&mut self, receiver: fn(String)) { 
        self.receivers.remove(&receiver);
    }

    pub fn dispatch_notify(&self, msg: String) { 
        for receiver in &self.receivers {
            receiver(msg.clone()); 
        }
    }
}

pub fn local_receiver_msg(msg: String) {
    println!("local receiver receive new msg {}", msg);
}

pub fn remote_receiver_msg(msg: String) {
    println!("remote receiver receive new msg {}", msg);
}

lazy_static::lazy_static! {
    pub static ref GPU_MSG: Mutex<MsgMgr> = Mutex::new(MsgMgr::new());
}

pub fn gpu_msg_register_receiver(receiver: fn(String)) {
    GPU_MSG.lock().unwrap().register_receiver(receiver);
}

pub fn gpu_msg_unregister_receiver(receiver: fn(String)) {
    GPU_MSG.lock().unwrap().unregister_receiver(receiver);
}

pub fn gpu_msg_dispatch_notify(msg: String) {
    GPU_MSG.lock().unwrap().dispatch_notify(msg);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observer_mode_test() {
        let mut observer = MsgMgr::new();
    
        observer.register_receiver(local_receiver_msg);
        observer.unregister_receiver(local_receiver_msg);
        observer.register_receiver(remote_receiver_msg);
        observer.dispatch_notify("hello world!".to_string());
    }
}