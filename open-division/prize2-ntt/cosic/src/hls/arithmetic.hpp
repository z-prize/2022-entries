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
#ifndef _ARITHMETIC_H_
#define _ARITHMETIC_H_

#include <ap_int.h>

#define _2_6 64
#define _2_12 4096
#define _2_18 262144
#define _2_24 16777216

typedef ap_uint<160> u160;
typedef ap_uint<128> u128;
typedef ap_uint<96> u96;
typedef ap_uint<65> u65;
typedef ap_uint<64> u64;
typedef ap_uint<33> u33;
typedef ap_uint<32> u32;
typedef ap_uint<6> u6;
typedef ap_uint<2> u2;


const u32 EPSILON = u32((1ull << 32) - 1);
const u64 MODULUS = u64(0xFFFFFFFF00000001);

u64 add(u64 left, u64 right);
u64 sub(u64 left, u64 right);
u64 mult(u64 left, u64 right);
void gs_butterfly(u64 *out_even, u64 *out_odd, u64 *in_even, u64 *in_odd, int exp);

#endif
