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

#include "arithmetic.hpp"

using namespace std;

// Add two Goldilocks elements together
//
// The number may not be between 0..MODULUS - 1,
// it can be between MODULUS - 1..2^64-1.
GF add(GF left, GF right) {
//#pragma HLS inline

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
//#pragma HLS inline

	GF sub, sub_offset, sub_temp, sub_nr_temp, DIF;
	ap_uint<1> c_sub, c_DIF;

	sub_temp = left - right; //perform initial subtraction

	c_sub = sub_temp.bit(64); //Bit Selection
	sub = sub_temp(63, 0);
	//        sub_offset = (c_sub==1)? M : 0;      //if the carry bit is 1 (negative value) Modulo offset is to be added

	if (c_sub) { sub_offset = MODULUS; }
	else { sub_offset = 0; }

	sub_nr_temp = sub + sub_offset;      //assign the output to the sum of the subtraction result and the offset
	c_DIF = sub_nr_temp.bit(64);
	DIF = sub_nr_temp(63, 0);

	return DIF;
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

GF to_canonical(GF x) {
  if (x >= MODULUS) {
    return x - MODULUS;
  }
  return x;
}
