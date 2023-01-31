/*
 * Copyright (C) 2022 DZK
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*******************************************************************************
 * Import Header File
 */
#include "ntt.hpp"
/*******************************************************************************
 * NTT 2^24
 */
void NTT_2_24_in_place(GF (&in)[16777216], GF powers[16777216], GF (&out)[16777216]) {
	// Variable declarations to avoid redeclaration
  	unsigned int m, k, i, j, u;
  	GF t, tmp, w;

  	// Declare n and logn for input length
  	const ap_uint<25> n = 16777216;
  	const int logn = 24;
//std::cout << "check 1" << std::endl;
  	// Declare and set internal computation array
  	static GF comp[16777216] = {};
  	for (int b = 0 ; b < n ; b++) { comp[b] = in[b]; }
  	//std::cout << "check 2" << std::endl;

  	// Cooley-Tukey loop, standard order input to bit-reverse order output
  	m = n;
  	for (k = 1; k < n; k <<= 1) {
  		// Maintain invariant s.t. n = m * 2k
  		m >>= 1;

  		// Loop over first of 2k blocks, no mult needed
  	    for (j = 0 ; j < m ; j++) {
  	    	t = comp[j + m];
  	    	tmp = comp[j];
  	    	comp[j + m] = sub(tmp, t);
  	    	comp[j] = add(tmp, t);
  	    }

  	    // Loop through remaining 2k-1 blocks
  	    u = 0;
  	    for (i = 1 ; i < k ; i++) {
  	    	// w is 2^k-th root of unity to the power of the bit-reverse of i
  	    	w = powers[k + i];

  	    	// Loop over current block of size m, starting from u
  	    	u += 2 * m;
  	    	for (j = u ; j < u + m ; j++) {
  	    		t = mult(comp[j + m], w);
  	  	    	tmp = comp[j];
  	    		comp[j + m] = sub(tmp, t);
  	    		comp[j] = add(tmp, t);
  	    	}
  	    }
  	}

  	// Form output
  	for (int i = 0 ; i < n ; i++) { out[i] = comp[i]; }

}
