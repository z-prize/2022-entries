// struct poly_t_matrix_t {
//     size_t    len;
//     uint32_t* r;
//     uint32_t* c;
//     fr_t*     coeff;
// };
// struct poly_t_cache_t {
//     poly_t_matrix_t a;
//     poly_t_matrix_t b;
//     poly_t_matrix_t c;
// };

// compute work schedule
// - (out_idx, matrix_id, coeff_idx)
// sort work
// RLE
// create tasks

enum poly_t_eta {
                 ETA_A = 0,
                 ETA_B,
                 ETA_C
};
struct poly_t_work_item_t {
    uint32_t out_idx;
    uint32_t r_idx;
    uint32_t coeff_idx;
    poly_t_eta eta;
};
int poly_t_work_item_t_cmp(const void* _a, const void* _b) {
    poly_t_work_item_t* a = (poly_t_work_item_t*)_a;
    poly_t_work_item_t* b = (poly_t_work_item_t*)_b;
    return((int)a->out_idx - (int)b->out_idx);
}

struct poly_t_idx_task_t {
    uint32_t out_idx;
    uint32_t len;
    uint32_t work_idx; // index into poly_t_work_item_t array
};
int poly_t_idx_task_t_cmp(const void* _a, const void* _b) {
    poly_t_idx_task_t* a = (poly_t_idx_task_t*)_a;
    poly_t_idx_task_t* b = (poly_t_idx_task_t*)_b;
    return((int)b->len - (int)a->len);
}

struct poly_t_final_sum_map_t {
    uint32_t task_idx;
    uint32_t out_idx;
};
int poly_t_final_sum_map_cmp(const void* _a, const void* _b) {
    poly_t_final_sum_map_t* a = (poly_t_final_sum_map_t*)_a;
    poly_t_final_sum_map_t* b = (poly_t_final_sum_map_t*)_b;
    return((int)a->out_idx - (int)b->out_idx);
}

struct poly_t_final_start_count_t {
    uint32_t count;
    uint32_t start;
};
int poly_t_final_start_count_cmp(const void* _a, const void* _b) {
    poly_t_final_start_count_t* a = (poly_t_final_start_count_t*)_a;
    poly_t_final_start_count_t* b = (poly_t_final_start_count_t*)_b;
    return((int)b->count - (int)a->count);
}

struct poly_t_cache_t {
    uint32_t lg_constraint_domain_size;
    uint32_t lg_input_domain_size;
    uint32_t work_len;
    fr_t*    coeffs;
    fr_t*    d_coeffs;
    poly_t_work_item_t* work;
    poly_t_work_item_t* d_work;

    uint32_t task_count;
    poly_t_idx_task_t*  tasks;
    poly_t_idx_task_t*  d_tasks;

    uint32_t final_nonzero_count;
    poly_t_final_sum_map_t* final_sum_map;
    poly_t_final_sum_map_t* d_final_sum_map;
    poly_t_final_start_count_t *final_start_count;
    poly_t_final_start_count_t *d_final_start_count;
};

void compute_poly_t_work_matrix(poly_t_work_item_t *work, size_t a_len, uint32_t* a_r, uint32_t* a_c,
                                uint32_t coeff_offset, uint32_t input_domain_size, uint32_t lg_period,
                                poly_t_eta eta) {
    // Compute a work
    for(size_t idx = 0; idx < a_len; idx++) {
        uint32_t r = a_r[idx];
        uint32_t c = a_c[idx];
        uint32_t index;
        if (c < input_domain_size) {
            index = c << lg_period;
        } else {
            uint32_t i = c - input_domain_size;
            uint32_t x = (1 << lg_period) - 1;
            index = i + (i / x) + 1;
        }
        work[idx].out_idx   = index;
        work[idx].r_idx     = r;
        work[idx].coeff_idx = idx + coeff_offset;
        work[idx].eta       = eta;
    }
}

void compute_poly_t_work(poly_t_cache_t& cache,
                         size_t a_len, uint32_t* a_r, uint32_t* a_c, fr_t* a_coeff,
                         size_t b_len, uint32_t* b_r, uint32_t* b_c, fr_t* b_coeff,
                         size_t c_len, uint32_t* c_r, uint32_t* c_c, fr_t* c_coeff
                         ) {
    uint32_t lg_constraint_domain_size = 15;
    uint32_t lg_input_domain_size = 2;
    uint32_t lg_period = lg_constraint_domain_size - lg_input_domain_size;
    uint32_t input_domain_size = 1 << lg_input_domain_size;
    uint32_t constraint_domain_size = 1 << lg_constraint_domain_size;
    
    size_t work_len = a_len + b_len + c_len;
    poly_t_work_item_t *work = new poly_t_work_item_t[work_len];

    compute_poly_t_work_matrix(work,                 a_len, a_r, a_c, 0,
                               input_domain_size, lg_period, ETA_A);
    compute_poly_t_work_matrix(&work[a_len],         b_len, b_r, b_c, a_len,
                               input_domain_size, lg_period, ETA_B);
    compute_poly_t_work_matrix(&work[a_len + b_len], c_len, c_r, c_c, a_len + b_len,
                               input_domain_size, lg_period, ETA_C);

    // Sort by out_idx
    qsort(work, work_len, sizeof(poly_t_work_item_t), poly_t_work_item_t_cmp);

    // for (size_t i = 0; i < work_len; i++) {
    //     printf("%ld: %5d\n", work[i].out_idx, i);
    // }

    // Count the number of tasks needed to update the output by index
    uint32_t max_size = 60;
    char *env_str = getenv("POLY_T_MAX_TASK");
    if (env_str) {
        max_size = strtol(env_str, NULL, 10);
        printf("Setting poly_t max size to %d\n", max_size);
    }
    
    uint32_t task_count = 0;
    uint32_t cur_index = work[0].out_idx;
    uint32_t cur_size = 0;
    for (size_t i = 0; i < work_len; i++) {
        if (work[i].out_idx != cur_index || cur_size == max_size) {
            task_count++;
            cur_size = 1;
            cur_index = work[i].out_idx;
        } else {
            cur_size++;
        }
    }
    task_count++;
    //printf("task_count %d\n", task_count);

    // Create tasks
    poly_t_idx_task_t* idx_tasks = new poly_t_idx_task_t[task_count];
    task_count = 0;
    cur_index = work[0].out_idx;
    cur_size = 0;
    for (size_t i = 0; i < work_len; i++) {
        if (work[i].out_idx != cur_index || cur_size == max_size) {
            idx_tasks[task_count].out_idx = cur_index;
            idx_tasks[task_count].len = cur_size;
            idx_tasks[task_count].work_idx = i - cur_size;
            
            task_count++;
            cur_size = 1;
            cur_index = work[i].out_idx;
        } else {
            cur_size++;
        }
    }
    idx_tasks[task_count].out_idx = cur_index;
    idx_tasks[task_count].len = cur_size;
    idx_tasks[task_count].work_idx = work_len - cur_size;
    task_count++;

    // printf("task list\n");
    // uint32_t item_count = 0;
    // for (size_t i = 0; i < task_count; i++) {
    //     printf("%5d: ", idx_tasks[i].out_idx);
    //     for (size_t j = 0; j < idx_tasks[i].len; j++) {
    //         printf("%d ", idx_tasks[i].work_idx + j);
    //     }
    //     printf("\n");
    //     item_count += idx_tasks[i].len;
    // }
    // printf("item_count %d\n", item_count);

    // Sort tasks by length, largest to smallest
    qsort(idx_tasks, task_count, sizeof(poly_t_idx_task_t), poly_t_idx_task_t_cmp);

    // printf("task list by length\n");
    // item_count = 0;
    // for (size_t i = 0; i < task_count; i++) {
    //     printf("%5d: ", idx_tasks[i].out_idx);
    //     for (size_t j = 0; j < idx_tasks[i].len; j++) {
    //         printf("%d ", idx_tasks[i].work_idx + j);
    //     }
    //     printf("\n");
    //     item_count += idx_tasks[i].len;
    // }
    // printf("item_count %d\n", item_count);

    // Enable final summation by creating a list that maps the tasks to to polynomial
    // index.
    poly_t_final_sum_map_t* final_sum_map = new poly_t_final_sum_map_t[task_count];
    // Count the inputs to each output index
    poly_t_final_start_count_t* out_idx_count = new poly_t_final_start_count_t[constraint_domain_size];
    memset(out_idx_count, 0, sizeof(poly_t_final_start_count_t) * constraint_domain_size);
    for (size_t i = 0; i < task_count; i++) {
        final_sum_map[i].out_idx = idx_tasks[i].out_idx;
        final_sum_map[i].task_idx = i;
        out_idx_count[idx_tasks[i].out_idx].count++;
    }
    // Sort the final sum map by out index
    qsort(final_sum_map, task_count, sizeof(poly_t_final_sum_map_t), poly_t_final_sum_map_cmp);

    // Determine starting offset for each output index
    uint32_t cur_offset = 0;
    uint32_t final_nonzero_count = 0;
    for (size_t i = 0; i < constraint_domain_size; i++) {
        out_idx_count[i].start = cur_offset;
        cur_offset += out_idx_count[i].count;
        if (out_idx_count[i].count > 0) {
            final_nonzero_count++;
        }
    }

    // Sort the final count/start by size
    qsort(out_idx_count, constraint_domain_size,
          sizeof(poly_t_final_start_count_t), poly_t_final_start_count_cmp);

    // printf("Final sum map\n");
    // for (size_t i = 0; i < task_count; i++) {
    //     printf("%5ld: %d -> %d\n", i,
    //            final_sum_map[i].task_idx, final_sum_map[i].out_idx);
    // }
    // // Note the vast majority are count 1. zero is the exception
    // printf("Final out count and offset\n");
    // for (size_t i = 0; i < constraint_domain_size; i++) {
    //     printf("%ld: %d %d\n", i, out_idx_count[i].count, out_idx_count[i].start);
    // }
    // printf("final non-zero count %d\n", final_nonzero_count);
    
    cache.work = work;
    cache.work_len = work_len;
    cache.tasks = idx_tasks;
    cache.task_count = task_count;
    cache.lg_constraint_domain_size = lg_constraint_domain_size;
    cache.lg_input_domain_size = lg_input_domain_size;
    cache.coeffs = new fr_t[work_len];
    cache.final_nonzero_count = final_nonzero_count;
    cache.final_sum_map = final_sum_map;
    cache.final_start_count = out_idx_count;
    memcpy(cache.coeffs,                 a_coeff, sizeof(fr_t) * a_len);
    memcpy(&cache.coeffs[a_len],         b_coeff, sizeof(fr_t) * b_len);
    memcpy(&cache.coeffs[a_len + b_len], c_coeff, sizeof(fr_t) * c_len);
}

#ifndef __CUDA_ARCH__
#if zero
void compute_poly_t_host(poly_t_cache_t cache,
                         fr_t* d_out, fr_t* d_poly,
                         uint32_t lg_constraint_domain_size, uint32_t lg_input_domain_size,
                         uint32_t lg_period) {
    assert(cache.lg_constraint_domain_size == lg_constraint_domain_size);
    assert(cache.lg_input_domain_size == lg_input_domain_size);
    uint32_t constraint_domain_size = 1 << lg_constraint_domain_size;
    memset((uint8_t*)d_out, 0, sizeof(fr_t) * constraint_domain_size);

    fr_t* eta_b = d_poly;
    fr_t* eta_c = &d_poly[1];
    fr_t* r_alpha_x_evals = &d_poly[2];

    // printf("eta_b ");
    // host_print_fr(*eta_b);

    // for (size_t i = 0; i < cache.work_len; i++) {
    //     poly_t_work_item_t &work = cache.work[i];
    //     if (work.eta == ETA_A) {
    //         d_out[work.out_idx] += cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
    //     } else if (work.eta == ETA_B) {
    //         d_out[work.out_idx] += *eta_b * cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
    //     } else if (work.eta == ETA_C) {
    //         d_out[work.out_idx] += *eta_c * cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
    //     }
    // }

    // for (size_t i = 0; i < cache.task_count; i++) {
    //     uint32_t out_idx = cache.tasks[i].out_idx;
    //     for (size_t j = 0; j < cache.tasks[i].len; j++) {
    //         poly_t_work_item_t &work = cache.work[cache.tasks[i].work_idx + j];
    //         if (work.eta == ETA_A) {
    //             d_out[out_idx] += cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
    //         } else if (work.eta == ETA_B) {
    //             d_out[out_idx] += *eta_b * cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
    //         } else if (work.eta == ETA_C) {
    //             d_out[out_idx] += *eta_c * cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
    //         }
    //     }
    // }

    // Create subtotal for each task
    fr_t* task_totals = new fr_t[cache.task_count];
    for (size_t i = 0; i < cache.task_count; i++) {
        task_totals[i].zero();
        for (size_t j = 0; j < cache.tasks[i].len; j++) {
            poly_t_work_item_t &work = cache.work[cache.tasks[i].work_idx + j];
            if (work.eta == ETA_A) {
                task_totals[i] += cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
            } else if (work.eta == ETA_B) {
                task_totals[i] += *eta_b * cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
            } else if (work.eta == ETA_C) {
                task_totals[i] += *eta_c * cache.coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
            }
        }
    }

    // Final summation
    // for (size_t i = 0; i < cache.task_count; i++) {
    //     uint32_t out_idx = cache.tasks[i].out_idx;
    //     d_out[out_idx] += task_totals[i];
    // }
    for (size_t i = 0; i < cache.final_nonzero_count; i++) {
        uint32_t start = cache.final_start_count[i].start;
        uint32_t count = cache.final_start_count[i].count;
        uint32_t out_idx = cache.final_sum_map[start].out_idx;
        for (size_t j = 0; j < count; j++) {
            uint32_t task_idx = cache.final_sum_map[start + j].task_idx;
            d_out[out_idx] += task_totals[task_idx];
        }
    }
    delete [] task_totals;
}
#endif
#endif

__global__
void compute_poly_t_partial(poly_t_cache_t cache,
                            uint32_t *atomic_task_count,
                            fr_t* d_out, fr_t* d_poly,
                            uint32_t lg_constraint_domain_size, uint32_t lg_input_domain_size,
                            uint32_t lg_period) {
#ifdef __CUDA_ARCH__
    int32_t  warp_thread            = threadIdx.x & 0x1F;
    //uint32_t constraint_domain_size = 1 << lg_constraint_domain_size;

    fr_t* eta_b = d_poly;
    fr_t* eta_c = &d_poly[1];
    fr_t* r_alpha_x_evals = &d_poly[2];
    
    while (true) {
        uint32_t task;
        if (warp_thread == 0) {
            task = atomicAdd(atomic_task_count, 32);
        }
        task = __shfl_sync(0xFFFFFFFF, task, 0);
        //printf("task=%d\n", task);
        
        uint32_t task_idx = task + warp_thread;

        if (task_idx >= cache.task_count)
            break;
    
        fr_t sum;
        sum.zero();
        for (size_t j = 0; j < cache.d_tasks[task_idx].len; j++) {
            poly_t_work_item_t &work = cache.d_work[cache.d_tasks[task_idx].work_idx + j];
            if (work.eta == ETA_A) {
                sum += cache.d_coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
            } else if (work.eta == ETA_B) {
                sum += *eta_b * cache.d_coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
            } else if (work.eta == ETA_C) {
                sum += *eta_c * cache.d_coeffs[work.coeff_idx] * r_alpha_x_evals[work.r_idx];
            }
        }
        d_out[task_idx] = sum;
    }
    
#endif
}
__global__
void compute_poly_t_final(poly_t_cache_t cache,
                          fr_t* d_out, fr_t* d_partial_sums) {
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    if (idx > cache.final_nonzero_count) {
        return;
    }

    uint32_t start = cache.d_final_start_count[idx].start;
    uint32_t count = cache.d_final_start_count[idx].count;
    uint32_t out_idx = cache.d_final_sum_map[start].out_idx;

    for (size_t j = 0; j < count; j++) {
        uint32_t task_idx = cache.d_final_sum_map[start + j].task_idx;
        d_out[out_idx] += d_partial_sums[task_idx];
    }
}
