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
#ifndef _NTT_H_
#define _NTT_H_

#include "arithmetic.hpp"
#include "omega_2_6.hpp"
#include "omega_2_12.hpp"

#include <hls_vector.h>
#include <hls_stream.h>

typedef hls::vector<u64, 64> v64u64;
typedef hls::vector<u64, 4> v4u64;

v64u64 ntt_2_6(v64u64 in);

void ntt_2_12_dataflow(hls::stream<v64u64> &out, hls::stream<v64u64> &in);
void ntt_2_12_serial(hls::stream<v64u64> &out, hls::stream<v64u64> &in);

v64u64 xor_permute_spatial(v64u64 in, ap_uint<6> offset);

extern "C"
{
    void ntt_2_24(u64 *out0, u64 *out1, u64 *out2, u64 *out3, u64 *out4, u64 *out5, u64 *out6, u64 *out7, u64 *out8, u64 *out9, u64 *out10, u64 *out11, u64 *out12, u64 *out13, u64 *out14, u64 *out15, u64 *in0, u64 *in1, u64 *in2, u64 *in3, u64 *in4, u64 *in5, u64 *in6, u64 *in7, u64 *in8, u64 *in9, u64 *in10, u64 *in11, u64 *in12, u64 *in13, u64 *in14, u64 *in15);
}

#endif
