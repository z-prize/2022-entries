/* MIT License

Copyright (c) 2022-2023 COSIC KU Leuven, 3001 Leuven, Belgium
Authors: Michiel Van Beirendonck <michiel.vanbeirendonck@esat.kuleuven.be>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */
#include "arithmetic.hpp"

// Add two u64 elements together
//
// The number MUST be between 0..MODULUS - 1,
// it cannot be between MODULUS - 1..2^64-1.
u64 add(u64 left, u64 right)
{
#pragma HLS inline

	u65 res = left + right;

	bool overflow = res.get_bit(64);
	res.set_bit(64, false);
	if (overflow)
	{
		res = res + EPSILON;
	}

	return res.range(63, 0);
}

// Modular reduction for u96, u128, and u160
template <int T>
u64 reduce(ap_uint<T> in)
{
#pragma HLS inline

	u32 e = (T > 128) ? in.range(159, 128) : 0;
	u32 a = (T > 96) ? in.range(127, 96) : 0;
	u32 b = in.range(95, 64);
	u32 c = in.range(63, 32);
	u32 d = in.range(31, 0);

	u2 dab_s, bc_s, s;
	u32 dab, bc;
	u64 res;

	u33 ab = a + b;
	(dab_s, dab) = d - ab;
	s[1] = (dab_s[1] & ~dab_s[0]);
	s[0] = (dab_s[1] & dab_s[0]);
	(bc_s, bc) = b + c - e - s;

	if (bc_s[0])
	{
		if (bc_s[1])
			res = (bc, dab) - EPSILON;
		else
			res = (bc, dab) + EPSILON;
	}
	else
	{
		res = (bc, dab);
	}

	return res;
}

// Multiply a u64 with omega_64^exp. This exploits that omega_64 = 2^3
template <class T>
u64 mult_twiddle(T left, int exp)
{
#pragma HLS function_instantiate variable = exp
#pragma HLS inline recursive

	// Note that these branches are evaluated at "compile-time", and not present in the circuit
	if (exp > 21)
	{
		u160 shifted = ((u160)left) << (3 * exp);
		return reduce(shifted);
	}
	else if (exp > 10)
	{
		u128 shifted = ((u128)left) << (3 * exp);
		return reduce(shifted);
	}
	else
	{
		u96 shifted = ((u96)left) << (3 * exp);
		return reduce(shifted);
	}
}

// Multiply two u64
u64 mult(u64 left, u64 right)
{
#pragma HLS inline recursive

	u128 res = left * right;
	return reduce(res);
}

// Gentleman-Sande NTT butterfly
void gs_butterfly(u64 *out_even, u64 *out_odd, u64 *in_even, u64 *in_odd, int exp)
{
#pragma HLS function_instantiate variable = exp
#pragma HLS inline recursive

	// We do not reduce tmp1 to u64; it will be reduced by multTwiddle64, which can accept u65 inputs
	u64 tmp0 = add(*in_even, *in_odd);
	u65 tmp1 = (u65)(*in_even) - (u65)(*in_odd) + (u65)MODULUS;
	*out_even = tmp0;
	*out_odd = mult_twiddle<u65>(tmp1, exp);
}
