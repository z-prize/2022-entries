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
#include "ntt.hpp"

// One of the 6 stages in an NTT_2_6
v64u64 ntt_2_6_stage(v64u64 in, int stage)
{
#pragma HLS function_instantiate variable = stage
	const int N = 64;
	int stride = (N >> (stage + 1));
	int exp_incr = 1 << (stage);
	v64u64 out;

stage:
	for (size_t i = 0; i < N / 2; i++)
	{
#pragma HLS unroll
		int even = (i % stride) + (i / stride) * 2 * stride;
		int odd = even + stride;
		int exp = (i % stride) * exp_incr;

		if (stage == 5)
		{
			// bit-reverse outputs
			gs_butterfly(&out[((ap_uint<6>)even).reverse()], &out[((ap_uint<6>)odd).reverse()], &in[even], &in[odd], exp);
		}
		else
		{
			gs_butterfly(&out[even], &out[odd], &in[even], &in[odd], exp);
		}
	}

	return out;
}

v64u64 ntt_2_6(v64u64 in)
{
	const int log2N = 6;
	v64u64 stages[log2N + 1];
#pragma HLS array_partition variable = stages type = complete dim = 0
#pragma HLS pipeline II = 1

	stages[0] = in;

ntt_2_6_stages:
	for (size_t i = 0; i < log2N; i++)
	{
#pragma HLS unroll
		stages[i + 1] = ntt_2_6_stage(stages[i], i);
	}

	return stages[log2N];
}
