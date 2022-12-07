/// GpuPolyContainer provides a space to keep all the allocated GPU memory between different functions
/// The biggest advantage is saving cuMemAlloc and cuMemFree
/// Since Cuda Context are created per process (actually), so as Container; it can be shared between threads which correspond to the same context

/// named buffers are used for saving polynomials
/// tmp buffers are used for reserving allocated gpu memory spaces
use crate::params::*;
use crate::{GPUError, GPUResult, GpuPoly, PolyKernel, PrimeField};
use crypto_cuda::DeviceMemory;

use std::collections::BTreeMap;

// use log::*;

pub struct GpuPolyContainer<F>
where
    F: PrimeField,
{
    named_buffers: BTreeMap<String, DeviceMemory<F>>,
    tmp_buffers: BTreeMap<usize, Vec<DeviceMemory<F>>>,
}

impl<F> GpuPolyContainer<F>
where
    F: PrimeField,
{
    pub fn create() -> GPUResult<Self> {
        let named_buffers = BTreeMap::new();
        let tmp_buffers = BTreeMap::new();

        Ok(Self {
            named_buffers,
            tmp_buffers,
        })
    }

    pub fn info(&self) -> GPUResult<()> {
        println!("");
        println!("===============    container tmp buffers    ===============");

        for (k, v) in self.tmp_buffers.iter() {
            println!("container has {} buffers of [{}] elements", v.len(), k);
            for i in v.iter() {
                println!("\t\t buffer address = {:x}", i.get_inner());
            }
        }

        println!("===============    container named buffers    ===============");

        for (k, v) in self.named_buffers.iter() {
            println!(
                "buffer |{}| at {:x} has [{}] elements",
                k,
                v.get_inner(),
                v.size()
            );
        }
        println!("===============    container   ===============");
        println!("");

        Ok(())
    }

    /// `ask_for` to get a specific length of gpu memory from the container
    /// if there is no available DeviceMemory that can be utilized, use `kern` to allocate a new one
    pub fn ask_for<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
        n: usize,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        if let Some(v) = self.tmp_buffers.get_mut(&n) {
            if let Some(b) = v.pop() {
                return GpuPoly::<F>::new_from_memory(kern, b);
            }
        }

        // we should allocate a gpu buffer space in case we don't have a reserved one
        let gpu_poly = GpuPoly::new(kern, n)?;

        Ok(gpu_poly)
    }

    /// for convenience
    pub fn ask_for_max<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        self.ask_for(kern, MAX_POLY_LEN)
    }

    /// for convenience
    pub fn ask_for_powers<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        self.ask_for(kern, MAX_POLY_LEN_LOG)
    }

    /// for convenience
    pub fn ask_for_results<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        self.ask_for(kern, kern.get_core_x_batch())
    }

    /// for convenience
    pub fn ask_for_buckets<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        self.ask_for(kern, kern.get_core_x_batch() * MAX_BUCKET_NUM)
    }

    /// for convenience
    pub fn ask_for_pq<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        self.ask_for(kern, 1 << MAX_RADIX_DEGREE >> 1)
    }

    /// for convenience
    pub fn ask_for_omegas<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        self.ask_for(kern, LOG2_MAX_ELEMENTS)
    }

    pub fn ask_for_omegas_with_size<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
        n: usize,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        self.ask_for(kern, n)
    }

    /// `recycle` means not dropping the polynomial buffer, but take control of it for future reuse
    pub fn recycle<'a, 'b>(&mut self, gpu_poly: GpuPoly<'a, 'b, F>) -> GPUResult<()> {
        let n = gpu_poly.size();

        let memory = gpu_poly.take_memory();

        if let Some(v) = self.tmp_buffers.get_mut(&n) {
            v.push(memory);
        } else {
            let mut v = Vec::new();
            v.push(memory);

            self.tmp_buffers.insert(n, v);
        }

        Ok(())
    }

    /// `recycle` means not dropping the polynomial buffer, but take control of it for future reuse
    fn recycle_memory<'a, 'b>(&mut self, mem: DeviceMemory<F>) -> GPUResult<()> {
        let n = mem.size();

        if let Some(v) = self.tmp_buffers.get_mut(&n) {
            v.push(mem);
        } else {
            let mut v = Vec::new();
            v.push(mem);

            self.tmp_buffers.insert(n, v);
        }

        Ok(())
    }

    /// `find` a named polynomial in the container
    pub fn find<'a, 'b>(
        &mut self,
        kern: &'a PolyKernel<'b, F>,
        name: &str,
    ) -> GPUResult<GpuPoly<'a, 'b, F>> {
        let res = if let Some(memory) = self.named_buffers.remove(name) {
            let gpu_poly = GpuPoly::new_from_memory(kern, memory)?;

            Ok(gpu_poly)
        } else {
            Err(GPUError::Simple(format!(
                "Cannot find gpu buffer |{}|",
                name
            )))
        };

        res
    }

    /// `save` means you want to keep a named polynomial in the container
    /// if there is already a buffer with the same name, we ensure that the two buffers must have the same length
    /// then a memcpy is performed, and the newer `gpu_poly` is recycled
    pub fn save<'a, 'b>(&mut self, name: &str, gpu_poly: GpuPoly<'a, 'b, F>) -> GPUResult<()> {
        if let Some(memory) = self.named_buffers.remove(name) {
            if memory.size() != gpu_poly.size() {
                return Err(GPUError::Simple(format!(
                    "existing buffer |{}| with different lengths, {} != {}",
                    name,
                    memory.size(),
                    gpu_poly.size()
                )));
            }

            let new_memory = gpu_poly.take_memory();

            self.named_buffers.insert(name.to_owned(), new_memory);

            self.recycle_memory(memory)?;

            return Ok(());
        }

        let memory = gpu_poly.take_memory();

        self.named_buffers.insert(name.to_owned(), memory);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::cuda::{GpuPoly, GpuPolyContainer, PolyKernel};
    use crate::{GPUResult, GPU_CUDA_CORES};

    use ark_bls12_377::Fr;
    use ark_std::{One, UniformRand, Zero};

    use log::info;

    const TEST_VEC_LEN: usize = 1 << 13;
    const GPU_DEV_IDX: usize = 0;

    #[test]
    fn init() {
        let _ = env_logger::try_init();
    }

    #[test]
    pub fn test_buffer_as_poly_container() -> GPUResult<()> {
        init();

        let mut rng = ark_std::test_rng();

        let poly_kern = PolyKernel::<Fr>::create_with_core(&GPU_CUDA_CORES[GPU_DEV_IDX])?;
        let mut gpu_poly_container = GpuPolyContainer::<Fr>::create()?;

        let test_vec = (0..TEST_VEC_LEN)
            .map(|_| Fr::rand(&mut rng))
            .collect::<Vec<_>>();

        let mut gpu_test_vec = vec![Fr::one(); TEST_VEC_LEN];

        info!("testing polynomial container [gpu]");

        let mut gpu_poly = GpuPoly::<Fr>::new(&poly_kern, TEST_VEC_LEN)?;
        info!(
            "gpu_poly.inner = {:x}, gpu_poly.size = {}",
            gpu_poly.get_memory().get_inner(),
            gpu_poly.size()
        );

        gpu_poly.fill_with_fe(&Fr::zero())?;
        gpu_poly.read_from(&test_vec)?;

        gpu_poly_container.save("test_01", gpu_poly)?;

        let gpu_poly_2 = gpu_poly_container.find(&poly_kern, "test_01")?;

        gpu_poly_2.write_to(gpu_test_vec.as_mut_slice())?;
        info!(
            "gpu_poly_2.inner = {:x}, gpu_poly_2.size = {}",
            gpu_poly_2.get_memory().get_inner(),
            gpu_poly_2.size()
        );

        assert_eq!(gpu_test_vec, test_vec);

        gpu_poly_container.recycle(gpu_poly_2)?;

        let gpu_poly_3 = gpu_poly_container.ask_for(&poly_kern, TEST_VEC_LEN)?;
        info!(
            "gpu_poly_3.inner = {:x}, gpu_poly_3.size = {}",
            gpu_poly_3.get_memory().get_inner(),
            gpu_poly_3.size()
        );

        gpu_poly_3.write_to(gpu_test_vec.as_mut_slice())?;
        assert_eq!(gpu_test_vec, test_vec);

        let gpu_poly_4 = gpu_poly_container.ask_for(&poly_kern, TEST_VEC_LEN - 1)?;
        info!(
            "gpu_poly_4.inner = {:x}, gpu_poly_4.size = {}",
            gpu_poly_4.get_memory().get_inner(),
            gpu_poly_4.size()
        );

        gpu_poly_container.recycle(gpu_poly_4)?;

        Ok(())
    }
}
