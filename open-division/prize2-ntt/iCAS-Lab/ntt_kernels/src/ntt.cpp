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

#include <ap_int.h>

typedef ap_uint<64> GF;

//typedef hls::vector<GF, 8> NTT_vec;

GF add(GF left, GF right);
GF sub(GF left, GF right);
GF negate(GF x);
GF mult(GF left, GF right);
GF to_canonical(GF x);


const GF EPSILON = GF((1ull << 32) - 1);
const GF ZERO = GF(0);
const GF ONE = GF(1);
const GF TWO = GF(2);
const GF MODULUS = GF(0xFFFFFFFF00000001);


typedef ap_uint<128> GF_MULT;
const GF_MULT MASK = GF_MULT(0xFFFFFFFFFFFFFFFF);



const GF OMEGA[33] = {
    GF(1ull),                     // for a domain of 2^0
    GF(18446744069414584320ull),  // for a domain of 2^1
    GF(281474976710656ull),       // for a domain of 2^2
    GF(18446744069397807105ull),  // for a domain of 2^3
    GF(17293822564807737345ull),  // for a domain of 2^4
    GF(70368744161280ull),        // for a domain of 2^5
    GF(549755813888ull),          // for a domain of 2^6
    GF(17870292113338400769ull),  // for a domain of 2^7
    GF(13797081185216407910ull),  // for a domain of 2^8
    GF(1803076106186727246ull),   // for a domain of 2^9
    GF(11353340290879379826ull),  // for a domain of 2^10
    GF(455906449640507599ull),    // for a domain of 2^11
    GF(17492915097719143606ull),  // for a domain of 2^12
    GF(1532612707718625687ull),   // for a domain of 2^13
    GF(16207902636198568418ull),  // for a domain of 2^14
    GF(17776499369601055404ull),  // for a domain of 2^15
    GF(6115771955107415310ull),   // for a domain of 2^16
    GF(12380578893860276750ull),  // for a domain of 2^17
    GF(9306717745644682924ull),   // for a domain of 2^18
    GF(18146160046829613826ull),  // for a domain of 2^19
    GF(3511170319078647661ull),   // for a domain of 2^20
    GF(17654865857378133588ull),  // for a domain of 2^21
    GF(5416168637041100469ull),   // for a domain of 2^22
    GF(16905767614792059275ull),  // for a domain of 2^23
    GF(9713644485405565297ull),   // for a domain of 2^24
    GF(5456943929260765144ull),   // for a domain of 2^25
    GF(17096174751763063430ull),  // for a domain of 2^26
    GF(1213594585890690845ull),   // for a domain of 2^27
    GF(6414415596519834757ull),   // for a domain of 2^28
    GF(16116352524544190054ull),  // for a domain of 2^29
    GF(9123114210336311365ull),   // for a domain of 2^30
    GF(4614640910117430873ull),   // for a domain of 2^31
    GF(1753635133440165772ull)    // for a domain of 2^32
};
using namespace std;

GF to_canonical(GF x) {
  if (x >= MODULUS) {
    return x - MODULUS;
  }
  return x;
}
// Add two Goldilocks elements together
//
// The number may not be between 0..MODULUS - 1,
// it can be between MODULUS - 1..2^64-1.
GF add(GF left, GF right) {
#pragma HLS inline

	GF sum, sum_nr, sum_nr_temp, sum_temp, SUM;
	ap_uint<1> c_sum, c_sum_nr, select;

	sum_temp = left + right;

	c_sum = sum_temp.test(64); //    Bit Selection
	sum = sum_temp(63, 0);

	sum_nr_temp = sum - MODULUS;
	c_sum_nr = sum_nr_temp.test(64); //    Bit Selection
	sum_nr = sum_nr_temp(63, 0);

	select = ~((~c_sum) & c_sum_nr);
	SUM = (select == 1)? sum_nr : sum;

	return SUM;
}

// Subtract a Goldilocks element from another one
//
// The number may not be between 0..MODULUS - 1,
// it can be between MODULUS - 1..2^64-1.
GF sub(GF left, GF right) {
#pragma HLS inline

	GF sub, sub_offset, sub_temp, sub_nr_temp, subtract_res;
	ap_uint<1> c_sub, c_sres;

	sub_temp = left - right;

	c_sub = sub_temp.bit(64);
	sub = sub_temp(63, 0);


	if (c_sub) { sub_offset = MODULUS; }
	else { sub_offset = 0; }

	sub_nr_temp = sub + sub_offset;
	c_sres = sub_nr_temp.bit(64);
	subtract_res = sub_nr_temp(63, 0);

	return subtract_res;
}

// Negate a Goldilocks element
GF negate(GF x) {
  if (x.is_zero()) {
    return x;
  } else {
    return MODULUS - to_canonical(x);
  }
}

// Multiply two Goldilocks elements
GF mult(GF left, GF right) {
#pragma HLS inline
  ap_ulong left_u64 = left.to_uint64();
  ap_ulong right_u64 = right.to_uint64();

  GF_MULT left_u128 = GF_MULT(left_u64);
  GF_MULT right_u128 = GF_MULT(right_u64);

  GF_MULT res = left_u128 * right_u128;

  GF res_lo = GF((res & MASK).to_uint64());
  GF res_hi = GF((res >> 64).to_uint64());

  GF res_hi_hi = GF(res_hi >> 32);
  GF res_hi_lo = GF(res_hi & EPSILON);

  GF t0 = res_lo - res_hi_hi;
  bool underflow = t0.get_bit(64);
  t0.set_bit(64, false);
  if (underflow) {
    t0 = t0 - EPSILON;
  }

  GF t1 = res_hi_lo * EPSILON;

  GF final_res = t0 + t1;
  bool overflow = final_res.get_bit(64);
  final_res.set_bit(64, false);
  if (overflow) {
    final_res = final_res + EPSILON;
  }

  return final_res;
}

//


/*******************************************************************************
 * NTT 2^18
 */
/*
 extern "C" {
void NTT_2_24_in_place(GF (&in)[16777216], GF powers[16777216], hls::stream<GF>& out_stream) {
	// Variable declarations to avoid redeclaration
  	unsigned int m, k, i, j, u;
  	GF t, w;

  	// Declare n and logn for input length
  	const ap_uint<25> n = 16777216;
  	const int logn = 24;

  	// Cooley-Tukey loop, standard order input to bit-reverse order output
  	m = n;
  	for (k = 1; k < n; k <<= 1) {
  		// Maintain invariant s.t. n = m * 2k
  		m >>= 1;

  		// Loop over first of 2k blocks, no mult needed
  	    for (j = 0 ; j < m ; j++) {
  	    	t = in[j + m];
  	    	in[j + m] = sub(in[j], t);
  	    	in[j] = add(in[j], t);
  	    }

  	    // Loop through remaining 2k-1 blocks
  	    u = 0;
  	    for (i = 1 ; i < k ; i++) {
  	    	// w is 2^k-th root of unity to the power of the bit-reverse of i
  	    	w = powers[k + i];

  	    	// Loop over current block of size m, starting from u
  	    	u += 2 * m;
  	    	for (j = u ; j < u + m ; j++) {
  	    		t = mult(in[j + m], w);
  	    		in[j + m] = sub(in[j], t);
  	    		in[j] = add(in[j], t);
  	    	}
  	    }
  	}

  	// Form output stream
  	for (int i = 0 ; i < n ; i++) { out_stream.write(in[i]); }
}
}
*/

//extern "C" {
/*
void NTT_2_3_in_place(GF in[8], GF powers[8], GF obuf[8]) {

#pragma HLS interface mode=m_axi port=in bundle=gmem_in
#pragma HLS interface mode=m_axi port=powers bundle=gmem_powers
//#pragma HLS interface mode=m_axi port=obuf bundle=obuf

	// Variable declarations to avoid redeclaration
  	unsigned int m, k, i, j, u;
  	GF t, tmp, w;

  	// Declare n and logn for input length
  	const ap_uint<25> n = 8;
  	const int logn = 3;
  	// Declare and set internal computation array
  	      GF comp[8] = {};
  	      for (int b = 0 ; b < n ; b++) {
  	          comp[b] = in[b];
  	      }

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
  	      for (int i = 0 ; i < n ; i++) { obuf[i] = comp[i]; }
}
*/

/**** Here we change **/

void NTT_2_3_in_place(GF in[16777216], GF powers[16777216]){//, GF obuf[16777216]){//, GF testing[8], GF testing_2[8]) {
/*
#pragma HLS interface mode=m_axi port=in bundle=gmem_in
#pragma HLS interface mode=m_axi port=powers bundle=gmem_powers
#pragma HLS interface mode=m_axi port=testing bundle=test1
#pragma HLS interface mode=m_axi port=testing_2 bundle=test2
#pragma HLS interface mode=m_axi port=obuf bundle=result
*/

#pragma HLS INTERFACE mode=m_axi bundle=in max_widen_bitwidth=64 port=in
#pragma HLS INTERFACE mode=m_axi bundle=powers max_widen_bitwidth=64 port=powers

//#pragma HLS INTERFACE mode=m_axi bundle=obuf max_widen_bitwidth=64 port=obuf

//#pragma HLS INTERFACE mode=m_axi bundle=test1 max_widen_bitwidth=64 port=testing
//#pragma HLS INTERFACE mode=m_axi bundle=test2 max_widen_bitwidth=64 port=testing_2


	// Variable declarations to avoid redeclaration
  	unsigned int m, k, i, j, u;
  	GF t, tmp, w;
  //	GF int_arr[16777216];

  	// Declare n and logn for input length
  	const ap_uint<25> n = 16777216;
  	const int logn = 24;
  	//for (int i = 0 ; i < n ; i++) { obuf[i] = in[i]; }
  //	for (int i = 0 ; i < 8 ; i++) { int_arr[i] = in[i]+i; }
  	//for (int i = 0 ; i < 8 ; i++) { obuf[i] = powers[i]+i; }
  //	for (int i = 0 ; i < 8 ; i++) { testing[i] = in[i]; }
  //	for (int i = 0 ; i < 8 ; i++) { testing_2[i] = powers[i]; }
//}


  	// Cooley-Tukey loop, standard order input to bit-reverse order output
  	m = n;
  	for (k = 1; k < n; k <<= 1) {
  		// Maintain invariant s.t. n = m * 2k
  		m >>= 1;

  		// Loop over first of 2k blocks, no mult needed
  	    for (j = 0 ; j < m ; j++) {
  	    	t = in[j + m];
  	    	tmp = in[j];
  	    	in[j + m] = sub(tmp, t);
  	    	in[j] = add(tmp, t);
  	    }

  	    // Loop through remaining 2k-1 blocks
  	    u = 0;
  	    for (i = 1 ; i < k ; i++) {
  	    	// w is 2^k-th root of unity to the power of the bit-reverse of i
  	    	w = powers[k + i];

  	    	// Loop over current block of size m, starting from u
  	    	u += 2 * m;
  	    	for (j = u ; j < u + m ; j++) {
  	    		t = mult(in[j + m], w);
  	  	    	tmp = in[j];
  	  	    	in[j + m] = sub(tmp, t);
  	  	    	in[j] = add(tmp, t);
  	    	}
  	    }
  	}

  	// Form output
  //	for (int i = 0 ; i < n ; i++) { obuf[i] = in[i]; }
  	//for (int i = 0 ; i < n ; i++) { obuf[i] = powers[i]; }
}
//*/
/*
void NTT_2_3_in_place_optimized(GF (&in)[8], GF powers[8], hls::stream<GF>& out_stream) {
	// Variable declarations to avoid redeclaration
	unsigned int m, k, i, j, u;
	GF t, w;

	// Declare n and logn for input length
	const int n = 8;
	const int logn = 3;

	// Cooley-Tukey loop, standard order input to bit-reverse order output
	m = n;
	for (k = 1; k < n; k <<= 1) {
		// Maintain invariant s.t. n = m * 2k
		m >>= 1;

		// Loop over first of 2k blocks, no mult needed
	    for (j = 0 ; j < m ; j++) {
	    	t = in[j + m];
	    	in[j + m] = sub(in[j], t);
	    	in[j] = add(in[j], t);
	    }

	    // Loop through remaining 2k-1 blocks
	    u = 0;
	    for (i = 1 ; i < k ; i++) {
	    	// w is 2^k-th root of unity to the power of the bit-reverse of i
	    	w = powers[k + i];

	    	// Loop over current block of size m, starting from u
	    	u += 2 * m;
	    	for (j = u ; j < u + m ; j++) {
	    		t = mult(in[j + m], w);
	    		in[j + m] = sub(in[j], t);
	    		in[j] = add(in[j], t);
	    	}
	    }
	}

	// Form output stream
	for (int i = 0 ; i < n ; i++) { out_stream.write(in[i]); }
}
*/
//}




