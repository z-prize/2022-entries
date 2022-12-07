
//     a_arith_row_on_k: zeros     0, ones 27628
//     a_arith_col_on_k: zeros     0, ones 27628
// a_arith_row_col_on_k: zeros     0, ones 27628
//          a_arith_val: zeros     0, ones     0
//   a_arith_evals_on_k: zeros 27628, ones     0
//     b_arith_row_on_k: zeros     0, ones 27506
//     b_arith_col_on_k: zeros     0, ones 27505
// b_arith_row_col_on_k: zeros     0, ones 27505
//          b_arith_val: zeros     0, ones     0
//   b_arith_evals_on_k: zeros 27505, ones     0
//     c_arith_row_on_k: zeros     0, ones 17102
//     c_arith_col_on_k: zeros     0, ones 17102
// c_arith_row_col_on_k: zeros     0, ones 17102
//          c_arith_val: zeros     0, ones     0
//   c_arith_evals_on_k: zeros 17102, ones     0

enum ArithVar {
  ARITH_A = 0,
  ARITH_B,
  ARITH_C
};

struct arith_cache_arr_t {
  fr_t*  d_val;
  fr_t*  d_evals; // Will be bit reversed
  size_t skip_zeros;
  size_t skip_ones;
};

struct arith_cache_var_t {
  arith_cache_arr_t row_on_k;
  arith_cache_arr_t col_on_k;
  arith_cache_arr_t row_col_on_k;
  arith_cache_arr_t val;
  arith_cache_arr_t evals_on_k;
};

struct arith_cache_t {
  arith_cache_var_t vars[3];
};

__global__
void compute_matrix_sumcheck_step1(uint32_t domain_size, arith_cache_var_t cache, 
                                   fr_t* d_b_evals, fr_t* d_a_poly, 
                                   fr_t* d_inverses, fr_t* d_alpha, fr_t* d_beta,
                                   fr_t* d_alpha_beta, fr_t* d_v_H_alpha_v_H_beta) {
#ifdef __CUDA_ARCH__
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    //   b_evals = alpha_beta - alpha * row_on_k - beta * col_on_k + row_col_on_k
    fr_t d_tmp;
    fr_t b_eval;
    d_tmp          = *d_alpha * cache.row_on_k.d_evals[idx];
    b_eval         = *d_alpha_beta - d_tmp;
    d_tmp          = *d_beta * cache.col_on_k.d_evals[idx];
    b_eval         = b_eval - d_tmp;
    d_b_evals[idx] = b_eval + cache.row_col_on_k.d_evals[idx];
    
    if (idx < domain_size) {
      d_inverses[idx] = d_inverses[idx] * cache.evals_on_k.d_val[idx];
      // a_poly = arith_val * v_H_alpha_v_H_beta
      d_a_poly[idx] = cache.val.d_val[idx] * *d_v_H_alpha_v_H_beta;
    }
#endif
}
            

__global__
void compute_matrix_sumcheck_step1_1(uint32_t domain_size, arith_cache_var_t cache, 
                                     fr_t* d_b_evals, fr_t* d_a_poly, 
                                     fr_t* d_inverses, fr_t* d_alpha, fr_t* d_beta,
                                     fr_t* d_alpha_beta, fr_t* d_v_H_alpha_v_H_beta) {
#ifdef __CUDA_ARCH__
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    //   b_evals = alpha_beta - alpha * row_on_k - beta * col_on_k + row_col_on_k
    fr_t d_tmp;
    fr_t b_eval;
    d_tmp          = *d_alpha * cache.row_on_k.d_evals[idx];
    b_eval         = *d_alpha_beta - d_tmp;
    d_tmp          = *d_beta * cache.col_on_k.d_evals[idx];
    b_eval         = b_eval - d_tmp;
    d_b_evals[idx] = b_eval + cache.row_col_on_k.d_evals[idx];
    
    // if (idx < domain_size) {
    //   d_inverses[idx] = d_inverses[idx] * cache.evals_on_k.d_val[idx];
    //   // a_poly = arith_val * v_H_alpha_v_H_beta
    //   d_a_poly[idx] = cache.val.d_val[idx] * *d_v_H_alpha_v_H_beta;
    // }
#endif
}
            
__global__
void compute_matrix_sumcheck_step1_2(uint32_t domain_size, arith_cache_var_t cache, 
                                     fr_t* d_b_evals, fr_t* d_a_poly, 
                                     fr_t* d_inverses, fr_t* d_alpha, fr_t* d_beta,
                                     fr_t* d_alpha_beta, fr_t* d_v_H_alpha_v_H_beta) {
#ifdef __CUDA_ARCH__
    index_t idx = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    // //   b_evals = alpha_beta - alpha * row_on_k - beta * col_on_k + row_col_on_k
    // fr_t d_tmp;
    // fr_t b_eval;
    // d_tmp          = *d_alpha * cache.row_on_k.d_evals[idx];
    // b_eval         = *d_alpha_beta - d_tmp;
    // d_tmp          = *d_beta * cache.col_on_k.d_evals[idx];
    // b_eval         = b_eval - d_tmp;
    // d_b_evals[idx] = b_eval + cache.row_col_on_k.d_evals[idx];
    
    // if (idx < domain_size) {
      d_inverses[idx] = d_inverses[idx] * cache.evals_on_k.d_val[idx];
      // a_poly = arith_val * v_H_alpha_v_H_beta
      d_a_poly[idx] = cache.val.d_val[idx] * *d_v_H_alpha_v_H_beta;
    // }
#endif
}
            
