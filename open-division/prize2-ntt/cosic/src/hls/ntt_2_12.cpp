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

v64u64 xor_permute_spatial(v64u64 in, ap_uint<6> offset)
{
	const int log2N = 6;
	v64u64 stages[log2N + 1];
#pragma HLS array_partition variable = stages type = complete dim = 0
#pragma HLS pipeline II = 1

	stages[0] = in;

xor_permute_stages:
	for (size_t i = 0; i < log2N; i++)
	{
		for (size_t j = 0; j < 64; j++)
		{
			stages[i + 1][j] = stages[i][offset[i] ? j ^ (1 << i) : j];
		}
	}

	return stages[log2N];
}

void xor_permute_spatial_64x64(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
	for (size_t i = 0; i < 64; i++)
	{
#pragma HLS pipeline rewind
		out << xor_permute_spatial(in.read(), i);
	}
}

void xor_permute_temporal_64x64(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
	u64 bram[64][64];
#pragma HLS array_partition variable = bram type = complete dim = 2
#pragma HLS dataflow

	for (size_t i = 0; i < 64; i++)
	{
#pragma HLS pipeline rewind
		v64u64 v = in.read();

		for (size_t j = 0; j < 64; j++)
		{
#pragma HLS unroll
			bram[i][j] = v[j];
		}
	}

	for (size_t i = 0; i < 64; i++)
	{
#pragma HLS pipeline rewind
		v64u64 v;

		for (size_t j = 0; j < 64; j++)
		{
#pragma HLS unroll
			v[j] = bram[i ^ j][j];
		}

		out << v;
	}
}

void transpose_64x64(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
#pragma HLS dataflow
	hls::stream<v64u64> s[2];
	xor_permute_spatial_64x64(s[0], in);
	xor_permute_temporal_64x64(s[1], s[0]);
	xor_permute_spatial_64x64(out, s[1]);
}

void ntt_2_12_64x64(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
	for (size_t i = 0; i < 64; i++)
	{
#pragma HLS pipeline rewind
		out << ntt_2_6(in.read());
	}
}

void twiddle_mul_64x64(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
	for (size_t i = 0; i < 64; i++)
	{
#pragma HLS pipeline rewind

		v64u64 v = in.read();

		for (size_t j = 1; j < 64; j++)
		{
#pragma HLS unroll
			v[j] = mult(v[j], OMEGA_2_12[i * j]);
		}

		out << v;
	}
}

void ntt_2_12_dataflow(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
#pragma HLS dataflow
	hls::stream<v64u64> s[3];
	ntt_2_12_64x64(s[0], in);
	twiddle_mul_64x64(s[1], s[0]);
	transpose_64x64(s[2], s[1]);
	ntt_2_12_64x64(out, s[2]);
}

void ntt_2_12_serial(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
	u64 bram[64][64];

row_ntt_loop:
	for (int i = 0; i < 64; i++)
	{
#pragma HLS pipeline rewind

		v64u64 v;

	row_ntt:
		v = ntt_2_6(in.read());

	twiddle_mul:
		for (int j = 1; j < 64; j++)
		{
			v[j] = mult(v[j], OMEGA_2_12[i * j]);
		}

	bram_write:
		for (int j = 0; j < 64; j++)
		{
			bram[i][j] = v[j];
		}
	}

col_ntt_loop:
	for (int i = 0; i < 64; i++)
	{
#pragma HLS pipeline rewind

		v64u64 v;

	bram_read:
		for (int j = 0; j < 64; j++)
		{
			v[j] = bram[j][i];
		}

	col_ntt:
		v = ntt_2_6(v);

		out.write(v);
	}
}