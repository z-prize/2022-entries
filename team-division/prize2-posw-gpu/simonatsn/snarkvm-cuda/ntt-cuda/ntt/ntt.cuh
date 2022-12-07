#include <algorithm>
#include <cassert>

#ifndef __CUDA_ARCH__

# include <util/exception.cuh>
# include <util/rusterror.h>
# include <util/gpu_t.cuh>

#include <iostream>
#include <chrono>

#endif

#include "ntt_parameters/ntt_parameters.cuh"
#include "ntt_kernels/ntt_kernels.cu"

#ifndef __CUDA_ARCH__

template<class batch_ctx_t>
class batcher_t {
public:
    // Total batch size
    size_t N;
    // Size of work to schedule at a time
    size_t block_size;

    class helper_t {
        const gpu_t* _gpu;
        size_t _selector;
        channel_t<helper_t*>* _completions;

    public:
        helper_t(const gpu_t* gpu, size_t selector,
                 channel_t<helper_t*>* completions) :
            _gpu(gpu), _selector(selector), _completions(completions)
        {}
        const gpu_t& gpu() {
            _gpu->select();
            return *_gpu;
        }
        int selector() {
            return(_selector);
        }
        stream_t& stream() {
            return (*_gpu)[_selector];
        }
        void complete() {
            _completions->send(this);
        }
    };
    // Notify of completed work on a device
    static void cb(void* userData)
    {
        helper_t* ctx = reinterpret_cast<helper_t*>(userData);
        ctx->complete();
    }

    batch_ctx_t& ctx;
    std::vector<helper_t> helpers;
    channel_t<helper_t*> completions;

    batcher_t(const size_t _N, const size_t _block_size, batch_ctx_t& _ctx) :
        N(_N), block_size(_block_size), ctx(_ctx)
    {
        for (size_t j = 0; j < gpu_t::FLIP_FLOP; j++) {
            for (auto gpu: all_gpus()) {
                helpers.emplace_back(gpu, j, &completions);
            }
        }
    }

    void launch_block(batcher_t::helper_t &helper, size_t idx) {
        // Bound block_size by N
        size_t cur_block_size = std::min(block_size, N - idx);
        ctx.launch(helper, idx, cur_block_size);
    }

    void run() {
        size_t num_blocks = (N + block_size - 1) / block_size;
        size_t num_helpers = min(helpers.size(), num_blocks);
        size_t block = 0;

        // Start processing on each worker stream
        for (size_t w = 0; w < num_helpers && w < num_blocks; w++) {
            launch_block(helpers[w], w * block_size);
            block++;
        }
        size_t jobs_completed = 0;
        for (; block < num_blocks; block++) {
            // Wait for a worker to free up
            auto helper = completions.recv();
            jobs_completed++;
            launch_block(*helper, block * block_size);
        }
        // Wait for final jobs to complete
        while (jobs_completed < num_blocks) {
            completions.recv();
            jobs_completed++;
        }
    }
};

class NTT {
public:
    enum class InputOutputOrder { NN, NR, RN, RR };
    enum class Direction { forward, inverse };
    enum class Type { standard, coset };
    enum class Algorithm { GS, CT };

    // SNP: Make bit_rev available outside of NTT for polynomial operations
    static void bit_rev(fr_t* d_out, const fr_t* d_inp,
                        uint32_t lg_domain_size, stream_t& stream)
    {
        assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

        size_t domain_size = (size_t)1 << lg_domain_size;

        if (domain_size <= WARP_SZ)
            bit_rev_permutation
                <<<1, domain_size, 0, stream>>>
                (d_out, d_inp, lg_domain_size);
        else if (d_out == d_inp)
            bit_rev_permutation
                <<<domain_size/WARP_SZ, WARP_SZ, 0, stream>>>
                (d_out, d_inp, lg_domain_size);
        else if (domain_size < 512)
            bit_rev_permutation_aux
                <<<1, domain_size / 4, domain_size * sizeof(fr_t), stream>>>
                (d_out, d_inp, lg_domain_size);
        else
            bit_rev_permutation_aux
                <<<domain_size / 512, 128, 512 * sizeof(fr_t), stream>>>
                (d_out, d_inp, lg_domain_size);
    }

private:
    static void NTT_internal2(fr_t* d_inout, int lg_domain_size,
                              Direction direction, Algorithm algorithm,
                              stream_t& stream)
    {

        const bool intt = direction == Direction::forward ? false : true;
        const auto& ntt_parameters = *NTTParameters::all(intt)[stream];

        switch (algorithm) {
            case Algorithm::GS: {
                GS_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
            }
            case Algorithm::CT: {
                CT_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
            }
            default: {
                assert(false);
            }
        }
    }

    static void NTT_internal(fr_t* d_inout, uint32_t lg_domain_size,
                             InputOutputOrder order, Direction direction,
                             stream_t& stream)
    {
        // Pick an NTT algorithm based on the input order and the desired output
        // order of the data. In certain cases, bit reversal can be avoided which
        // results in a considerable performance gain.
        switch (order) {
            case InputOutputOrder::NN: {
                bit_rev(d_inout, d_inout, lg_domain_size, stream);
                NTT_internal2(d_inout, lg_domain_size, direction,
                              Algorithm::CT, stream);
                break;
            }
            case InputOutputOrder::NR: {
                NTT_internal2(d_inout, lg_domain_size, direction,
                              Algorithm::GS, stream);
                break;
            }
            case InputOutputOrder::RN: {
                NTT_internal2(d_inout, lg_domain_size, direction,
                              Algorithm::CT, stream);
                break;
            }
            case InputOutputOrder::RR: {
                NTT_internal2(d_inout, lg_domain_size, direction,
                              Algorithm::GS, stream);
                bit_rev(d_inout, d_inout, lg_domain_size, stream);
                break;
            }
            default: {
                assert(false);
            }
        }
    }

public:
    // Perform an NTT device data
    static void NTT_device(fr_t* d_inout, uint32_t lg_domain_size,
                           InputOutputOrder order, Direction direction,
                           stream_t& stream) {
        NTT_internal(d_inout, lg_domain_size, order, direction, stream);
    }
      
    static RustError Base(const gpu_t& gpu, fr_t* inout, uint32_t lg_domain_size,
                          InputOutputOrder order, Direction direction,
                          Type type)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> d_inout(domain_size);
            gpu.HtoD(&d_inout[0], inout, domain_size);

            NTT_internal(&d_inout[0], lg_domain_size, order, direction, gpu);

            gpu.DtoH(inout, &d_inout[0], domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    static RustError LDE(const gpu_t& gpu, fr_t* inout,
                         uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> d_inout(domain_size << lg_blowup);
            cudaMemsetAsync(&d_inout[domain_size], 0,
                sizeof(fr_t) * ((domain_size << lg_blowup) - domain_size),
                gpu);
            cudaDeviceSynchronize();
            gpu.HtoD(&d_inout[0], inout, domain_size);

            NTT_internal(&d_inout[0], lg_domain_size,
                         InputOutputOrder::NN, Direction::inverse, gpu);

            const fr_t* gen_powers =
                NTTParameters::all()[gpu.id()]->partial_group_gen_powers;

            if (domain_size < WARP_SZ)
                LDE_distribute_powers<<<1, domain_size, 0, gpu>>>
                    (&d_inout[0], lg_blowup, gen_powers);
            else if (domain_size < 1024)
                LDE_distribute_powers<<<domain_size / WARP_SZ, WARP_SZ, 0, gpu>>>
                    (&d_inout[0], lg_blowup, gen_powers);
            else
                LDE_distribute_powers<<<domain_size / 1024, 1024, 0, gpu>>>
                    (&d_inout[0], lg_blowup, gen_powers);

            NTT_internal(&d_inout[0], lg_domain_size + lg_blowup,
                         InputOutputOrder::NN, Direction::forward, gpu);

            gpu.DtoH(inout, &d_inout[0], domain_size << lg_blowup);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

private:
    static void LDE_launch(stream_t& stream, size_t kernel_sms,
                           fr_t* ext_domain_data, fr_t* domain_data,
                           const fr_t* gen_powers,
                           uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        assert(lg_domain_size + lg_blowup <= MAX_LG_DOMAIN_SIZE);
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size << lg_blowup;
        size_t device_max_threads = kernel_sms * 1024;
        uint32_t num_blocks, block_size;

        if (device_max_threads < domain_size) {
            num_blocks = kernel_sms;
            block_size = 1024;
        } else if (domain_size < 1024) {
            num_blocks = 1;
            block_size = domain_size;
        } else {
            num_blocks = domain_size / 1024;
            block_size = 1024;
        }

        stream.launch_coop(LDE_spread_distribute_powers,
                               num_blocks, block_size, sizeof(fr_t) * block_size,
                           ext_domain_data, domain_data, gen_powers,
                           lg_domain_size, lg_blowup);
    }

public:
    static RustError LDE_aux(const gpu_t& gpu, fr_t* inout,
                             uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        try {

            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size << lg_blowup;
            // The 2nd to last 'domain_size' chunk will hold the original data
            // The last chunk will get the bit reversed iNTT data
            dev_ptr_t<fr_t> d_inout(ext_domain_size + domain_size); // + domain_size for aux buffer
            cudaDeviceSynchronize();
            fr_t* aux_data = &d_inout[ext_domain_size];
            fr_t* domain_data = &d_inout[ext_domain_size - domain_size]; // aligned to the end
            fr_t* ext_domain_data = &d_inout[0];
            gpu.HtoD(domain_data, inout, domain_size);

            NTT_internal(domain_data, lg_domain_size,
                         InputOutputOrder::NR, Direction::inverse, gpu);

            const fr_t* gen_powers =
                NTTParameters::all()[gpu.id()]->partial_group_gen_powers;

            bit_rev(aux_data, domain_data, lg_domain_size, gpu);

            // Determine the max power of 2 SM count
            size_t kernel_sms = gpu.sm_count();
            while (kernel_sms & (kernel_sms - 1))
                kernel_sms -= (kernel_sms & (0 - kernel_sms));

            LDE_launch(gpu, kernel_sms, ext_domain_data, domain_data, gen_powers,
                       lg_domain_size, lg_blowup);

            // NTT - RN
            NTT_internal(ext_domain_data, lg_domain_size + lg_blowup,
                         InputOutputOrder::RN, Direction::forward, gpu);

            gpu.DtoH(inout, ext_domain_data, domain_size << lg_blowup);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

#if 0
    // |   stream0    |    stream1      |
    // +--------------+-----------------+
    // |  htod col0   |                 |
    // |   ntt col0   |   htod col1     |
    // |  dtoh col0   |    ntt col1     |
    // |              |   dtoh col1     |
    // |  htod col2   |                 |
    // |   ntt col2   |   htod col3     |
    // |  dtoh col2   |    ntt col3     |
    // |              |   dtoh col3     |
    //               etc
    static RustError Batch(const gpu_t& gpu, fr_t* inout, size_t N, uint32_t lg_domain_size,
                           InputOutputOrder order, Direction direction,
                           Type type)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            size_t domain_size = (size_t)1 << lg_domain_size;
#ifdef FLIPFLOP3
            dev_ptr_t<fr_t> d_inout(3 * domain_size);
            for (size_t i = 0; i < N; i++) {
                // Even work
                gpu[i].HtoD(&d_inout[(i%3) * domain_size], &inout[i * domain_size], domain_size);
                NTT_internal(&d_inout[(i%3) * domain_size],
                             lg_domain_size, order, direction, gpu[i]);
                gpu[i].DtoH(&inout[i * domain_size], &d_inout[(i%3) * domain_size], domain_size);
            }
            gpu[0].sync();
            gpu[1].sync();
            gpu[2].sync();
#else
            dev_ptr_t<fr_t> d_inout(2 * domain_size);
            for (size_t i = 0; i < N; i++) {
                // Even work
                gpu[i].HtoD(&d_inout[(i&1) * domain_size], &inout[i * domain_size], domain_size);
                NTT_internal(&d_inout[(i&1) * domain_size],
                             lg_domain_size, order, direction, gpu[i]);
                gpu[i].DtoH(&inout[i * domain_size], &d_inout[(i&1) * domain_size], domain_size);
            }
            gpu[0].sync();
            gpu[1].sync();
#endif

        } catch (const cuda_error& e) {
            std::cerr << "Caught exception" << e.what() << std::endl;
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }
#endif

    struct lde_batch_ctx_t {
        uint32_t lg_domain_size;
        uint32_t lg_blowup;
        InputOutputOrder ntt_order;
        Direction ntt_direction;
        // Type type; // TODO
        fr_t *in;
        fr_t *out;
        std::vector<dev_ptr_t<fr_t>*>& dmem_all_gpus;
        lde_batch_ctx_t(uint32_t _lg_domain_size,
                        uint32_t _lg_blowup,
                        InputOutputOrder _ntt_order,
                        Direction _ntt_direction,
                        fr_t *_out, fr_t *_in,
                        std::vector<dev_ptr_t<fr_t>*>& _dmem) : dmem_all_gpus(_dmem)
        {
            lg_domain_size = _lg_domain_size;
            ntt_order = _ntt_order;
            ntt_direction = _ntt_direction;
            lg_blowup = _lg_blowup;
            in = _in;
            out = _out;
        }

        void launch(batcher_t<lde_batch_ctx_t>::helper_t &ctx,
                    size_t idx, size_t block_size) {
            const gpu_t& gpu = ctx.gpu();
            stream_t& stream = ctx.stream();
            dev_ptr_t<fr_t> *dmem = dmem_all_gpus[stream];
            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size << lg_blowup;

            // domain_data will be aligned to the end
            // aux data sits after ext_domain_data
            //                          input      aux
            //                       |- domain -|
            // |---------- ext_domain ----------|- domain -|
            size_t num_elements   = ext_domain_size + domain_size;
            fr_t* ff_data         = &(*dmem)[ctx.selector() * num_elements];
            fr_t* ext_domain_data = ff_data;
            fr_t* domain_data     = &ff_data[ext_domain_size - domain_size];
            fr_t* aux_data        = &ff_data[ext_domain_size];

            // Perform the callback one before the last item
            size_t cb_count = min(block_size - 1, block_size - 2);

            const fr_t* gen_powers =
                NTTParameters::all()[stream]->partial_group_gen_powers;

            // Determine the max power of 2 SM count
            // TODO: can precompute per device
            size_t kernel_sms = gpu.sm_count();
            while (kernel_sms & (kernel_sms - 1))
                kernel_sms -= (kernel_sms & (0 - kernel_sms));

            for (size_t i = 0; i < block_size; i++) {
                size_t job_idx = idx + i;

                // Copy the input data
                stream.HtoD(domain_data, &in[job_idx * domain_size], domain_size);

                // Determine NTT or LDE
                if (lg_blowup == 0) {
                    NTT_internal(domain_data, lg_domain_size,
                                 ntt_order, ntt_direction, stream);
                    stream.DtoH(&out[job_idx * domain_size], domain_data, domain_size);
                } else {
                    // Performing LDE
                    NTT_internal(domain_data, lg_domain_size,
                                 InputOutputOrder::NR, Direction::inverse, stream);

                    bit_rev(aux_data, domain_data, lg_domain_size, stream);

                    // Pull the intermediate result back to the host
                    // Should be fine to do here if bandwidth is the limiter
                    stream.DtoH(&in[job_idx * domain_size], aux_data, domain_size);

                    LDE_launch(stream, kernel_sms, ext_domain_data, domain_data, gen_powers,
                               lg_domain_size, lg_blowup);

                    NTT_internal(ext_domain_data, lg_domain_size + lg_blowup,
                                 InputOutputOrder::RN, Direction::forward, stream);

                    // Copy the data back
                    stream.DtoH(&out[job_idx * ext_domain_size], ext_domain_data, ext_domain_size);
                }

                // Start processing the next block when almost done
                if (i == cb_count) {
                    CUDA_OK(cudaLaunchHostFunc(stream, batcher_t<lde_batch_ctx_t>::cb, &ctx));
                }
            }
        }
    };

    static RustError LDE_batch(fr_t* out, fr_t* in,
                               std::vector<dev_ptr_t<fr_t>*> &dmem,
                               size_t N, uint32_t lg_domain_size,
                               uint32_t lg_blowup)
    {
        try {
            const size_t block_size = 5;
            assert(dmem.size() == ngpus());

            lde_batch_ctx_t batch_ctx(lg_domain_size,
                                      lg_blowup,
                                      InputOutputOrder::NN, // Not used by LDE
                                      Direction::forward, // Not used by LDE
                                      out, in, dmem);
            batcher_t<lde_batch_ctx_t> batcher(N, block_size, batch_ctx);
            batcher.run();

            for (auto gpu: all_gpus())
                gpu->sync();
        } catch (const cuda_error& e) {
            std::cerr << "exception " << e.what() << std::endl;
            for (auto gpu: all_gpus())
                gpu->sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    static RustError Batch(fr_t* out, fr_t* in,
                           std::vector<dev_ptr_t<fr_t>*> &dmem,
                           size_t N, uint32_t lg_domain_size,
                           InputOutputOrder ntt_order, Direction ntt_direction)
    {
        try {
            const size_t block_size = 5;
            assert(dmem.size() == ngpus());

            lde_batch_ctx_t batch_ctx(lg_domain_size,
                                      0, ntt_order, ntt_direction,
                                      out, in, dmem);
            batcher_t<lde_batch_ctx_t> batcher(N, block_size, batch_ctx);
            batcher.run();

            for (auto gpu: all_gpus())
                gpu->sync();
        } catch (const cuda_error& e) {
            std::cerr << "exception " << e.what() << std::endl;
            for (auto gpu: all_gpus())
                gpu->sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }
};

#endif
