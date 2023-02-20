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
#include "omega_2_24.hpp"
#include <iostream>

void xor_permute_spatial_4096x4096(hls::stream<v64u64> &sout, hls::stream<u64> sin[64], bool skip)
{
	for (size_t i = 0; i < 64; i++)
	{
		for (size_t j = 0; j < 64; j++)
		{
			for (size_t k = 0; k < 64; k++)
			{
#pragma HLS pipeline rewind
				v64u64 v;

				for (size_t l = 0; l < 64; l++)
				{
#pragma HLS unroll
					sin[l] >> v[l];
				}

				sout << xor_permute_spatial(v, skip ? 0 : i);
			}
		}
	}
}

void xor_permute_spatial_4096x4096(hls::stream<u64> sout[64], hls::stream<u64> sin[64], bool skip)
{
	for (size_t i = 0; i < 64; i++)
	{
		for (size_t j = 0; j < 64; j++)
		{
			for (size_t k = 0; k < 64; k++)
			{
#pragma HLS pipeline rewind
				v64u64 v;

				for (size_t l = 0; l < 64; l++)
				{
#pragma HLS unroll
					sin[l] >> v[l];
				}

				v = xor_permute_spatial(v, skip ? 0 : i);

				for (size_t l = 0; l < 64; l++)
				{
#pragma HLS unroll
					sout[l] << v[l];
				}
			}
		}
	}
}

void mm2s(hls::stream<u64> s[4], u64 *mm, ap_uint<1> pass, int hbm_pc)
{
#pragma HLS function_instantiate variable = hbm_pc

	for (size_t i = 0; i < 64; i++)
	{
		for (size_t j = 0; j < 64; j++)
		{
			for (size_t k = 0; k < 64; k++)
			{
#pragma HLS pipeline rewind
				// read 4 values
				int rd_address0, rd_address1, rd_address2, rd_address3;
				if (pass == 0)
				{
					rd_address0 = _2_18 * 0 + _2_12 * k + 64 * i + j;
					rd_address1 = _2_18 * 1 + _2_12 * k + 64 * i + j;
					rd_address2 = _2_18 * 2 + _2_12 * k + 64 * i + j;
					rd_address3 = _2_18 * 3 + _2_12 * k + 64 * i + j;
				}
				else
				{
					rd_address0 = 4 * (_2_12 * (i ^ (4 * hbm_pc + 0)) + _2_6 * k + j) + 0;
					rd_address1 = 4 * (_2_12 * (i ^ (4 * hbm_pc + 1)) + _2_6 * k + j) + 1;
					rd_address2 = 4 * (_2_12 * (i ^ (4 * hbm_pc + 2)) + _2_6 * k + j) + 2;
					rd_address3 = 4 * (_2_12 * (i ^ (4 * hbm_pc + 3)) + _2_6 * k + j) + 3;
				}
				s[0] << mm[rd_address0];
				s[1] << mm[rd_address1];
				s[2] << mm[rd_address2];
				s[3] << mm[rd_address3];
			}
		}
	}
}

void s2mm(u64 *mm, hls::stream<u64> s[4], ap_uint<1> pass, int hbm_pc)
{
#pragma HLS function_instantiate variable = hbm_pc

	for (size_t i = 0; i < 64; i++)
	{
		for (size_t j = 0; j < 64; j++)
		{
			for (size_t k = 0; k < 64; k++)
			{
#pragma HLS pipeline rewind
				// read 4 values
				int wr_address0, wr_address1, wr_address2, wr_address3;
				if (pass == 0)
				{
					wr_address0 = 4 * (_2_12 * i + _2_6 * j + k) + 0;
					wr_address1 = 4 * (_2_12 * i + _2_6 * j + k) + 1;
					wr_address2 = 4 * (_2_12 * i + _2_6 * j + k) + 2;
					wr_address3 = 4 * (_2_12 * i + _2_6 * j + k) + 3;
				}
				else
				{
					wr_address0 = _2_18 * 0 + _2_12 * k + 64 * i + j;
					wr_address1 = _2_18 * 1 + _2_12 * k + 64 * i + j;
					wr_address2 = _2_18 * 2 + _2_12 * k + 64 * i + j;
					wr_address3 = _2_18 * 3 + _2_12 * k + 64 * i + j;
				}
				s[0] >> mm[wr_address0];
				s[1] >> mm[wr_address1];
				s[2] >> mm[wr_address2];
				s[3] >> mm[wr_address3];
			}
		}
	}
}

void ntt_2_12_4096x4096(hls::stream<v64u64> &out, hls::stream<v64u64> &in)
{
	for (size_t i = 0; i < 4096; i++)
	{
#pragma HLS dataflow
#ifdef NTT_2_12_DATAFLOW
		ntt_2_12_dataflow(out, in);
#else
		ntt_2_12_serial(out, in);
#endif
	}
}

void twiddle_mul_4096x4096(hls::stream<u64> out[64], hls::stream<v64u64> &in, bool skip)
{
	for (size_t i = 0; i < 64; i++)
	{
		for (size_t j = 0; j < 64; j++)
		{
			for (size_t k = 0; k < 64; k++)
			{
// match twiddle_mul II to ntt_2_12 II
#ifdef NTT_2_12_DATAFLOW
#pragma HLS pipeline rewind
#else
#pragma HLS pipeline rewind II = 32
#endif
				v64u64 v = in.read();
			mul:
				for (size_t l = 0; l < 64; l++)
				{
					ap_uint<12> row = 64 * i + j;
					ap_uint<12> col = 64 * l + k;
					ap_uint<24> rowTimesCol = skip ? (ap_uint<24>)0 : (row * col);
					u64 twiddle = mult(OMEGA_2_12[rowTimesCol.range(23, 12)], OMEGA_2_24[rowTimesCol.range(11, 0)]);
					out[l] << mult(v[l], twiddle);
				}
			}
		}
	}
}

extern "C"
{
	void ntt_2_24_pass(u64 *out0, u64 *out1, u64 *out2, u64 *out3, u64 *out4, u64 *out5, u64 *out6, u64 *out7, u64 *out8, u64 *out9, u64 *out10, u64 *out11, u64 *out12, u64 *out13, u64 *out14, u64 *out15, u64 *in0, u64 *in1, u64 *in2, u64 *in3, u64 *in4, u64 *in5, u64 *in6, u64 *in7, u64 *in8, u64 *in9, u64 *in10, u64 *in11, u64 *in12, u64 *in13, u64 *in14, u64 *in15, int pass)
	{
		hls::stream<v64u64> s[2];
		hls::stream<u64> s0[64];
		hls::stream<u64> s1[64];
		hls::stream<u64> s2[64];
		u64 *in[16] = {in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15};
		u64 *out[16] = {out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15};

#pragma HLS dataflow

		bool skip_xor_permute_0 = (pass == 0);
		bool skip_twiddle_mul = (pass == 1);
		bool skip_xor_permute_1 = (pass == 1);

	mm2s:
		for (size_t hbm_pc = 0; hbm_pc < 16; hbm_pc++)
		{
#pragma HLS unroll
			mm2s(&s0[4 * hbm_pc], in[hbm_pc], pass, hbm_pc);
		}

		xor_permute_spatial_4096x4096(s[0], s0, skip_xor_permute_0);
		ntt_2_12_4096x4096(s[1], s[0]);
		twiddle_mul_4096x4096(s1, s[1], skip_twiddle_mul);
		xor_permute_spatial_4096x4096(s2, s1, skip_xor_permute_1);

	s2mm:
		for (size_t hbm_pc = 0; hbm_pc < 16; hbm_pc++)
		{
#pragma HLS unroll
			s2mm(out[hbm_pc], &s2[4 * hbm_pc], pass, hbm_pc);
		}
	}

	void ntt_2_24(u64 *out0, u64 *out1, u64 *out2, u64 *out3, u64 *out4, u64 *out5, u64 *out6, u64 *out7, u64 *out8, u64 *out9, u64 *out10, u64 *out11, u64 *out12, u64 *out13, u64 *out14, u64 *out15, u64 *in0, u64 *in1, u64 *in2, u64 *in3, u64 *in4, u64 *in5, u64 *in6, u64 *in7, u64 *in8, u64 *in9, u64 *in10, u64 *in11, u64 *in12, u64 *in13, u64 *in14, u64 *in15)
	{
#pragma HLS interface mode = m_axi port = out0 bundle = gmem0
#pragma HLS interface mode = m_axi port = out1 bundle = gmem1
#pragma HLS interface mode = m_axi port = out2 bundle = gmem2
#pragma HLS interface mode = m_axi port = out3 bundle = gmem3
#pragma HLS interface mode = m_axi port = out4 bundle = gmem4
#pragma HLS interface mode = m_axi port = out5 bundle = gmem5
#pragma HLS interface mode = m_axi port = out6 bundle = gmem6
#pragma HLS interface mode = m_axi port = out7 bundle = gmem7
#pragma HLS interface mode = m_axi port = out8 bundle = gmem8
#pragma HLS interface mode = m_axi port = out9 bundle = gmem9
#pragma HLS interface mode = m_axi port = out10 bundle = gmem10
#pragma HLS interface mode = m_axi port = out11 bundle = gmem11
#pragma HLS interface mode = m_axi port = out12 bundle = gmem12
#pragma HLS interface mode = m_axi port = out13 bundle = gmem13
#pragma HLS interface mode = m_axi port = out14 bundle = gmem14
#pragma HLS interface mode = m_axi port = out15 bundle = gmem15
#pragma HLS interface mode = m_axi port = in0 bundle = gmem0
#pragma HLS interface mode = m_axi port = in1 bundle = gmem1
#pragma HLS interface mode = m_axi port = in2 bundle = gmem2
#pragma HLS interface mode = m_axi port = in3 bundle = gmem3
#pragma HLS interface mode = m_axi port = in4 bundle = gmem4
#pragma HLS interface mode = m_axi port = in5 bundle = gmem5
#pragma HLS interface mode = m_axi port = in6 bundle = gmem6
#pragma HLS interface mode = m_axi port = in7 bundle = gmem7
#pragma HLS interface mode = m_axi port = in8 bundle = gmem8
#pragma HLS interface mode = m_axi port = in9 bundle = gmem9
#pragma HLS interface mode = m_axi port = in10 bundle = gmem10
#pragma HLS interface mode = m_axi port = in11 bundle = gmem11
#pragma HLS interface mode = m_axi port = in12 bundle = gmem12
#pragma HLS interface mode = m_axi port = in13 bundle = gmem13
#pragma HLS interface mode = m_axi port = in14 bundle = gmem14
#pragma HLS interface mode = m_axi port = in15 bundle = gmem15
		ntt_2_24_pass(out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, 0);
		ntt_2_24_pass(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12, in13, in14, in15, out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13, out14, out15, 1);
	}
}
