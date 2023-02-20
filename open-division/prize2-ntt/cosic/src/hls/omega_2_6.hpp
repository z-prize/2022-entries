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
#ifndef _OMEGA_2_6_H_
#define _OMEGA_2_6_H_

#include "arithmetic.hpp"

// The powers of omega_64
const u64 OMEGA_2_6[33] = {
	u64(1ull),					  // w_64^0
	u64(8ull),					  // w_64^1
	u64(64ull),					  // w_64^2
	u64(512ull),				  // w_64^3
	u64(4096ull),				  // w_64^4
	u64(32768ull),				  // w_64^5
	u64(262144ull),				  // w_64^6
	u64(2097152ull),			  // w_64^7
	u64(16777216ull),			  // w_64^8
	u64(134217728ull),			  // w_64^9
	u64(1073741824ull),			  // w_64^10
	u64(8589934592ull),			  // w_64^11
	u64(68719476736ull),		  // w_64^12
	u64(549755813888ull),		  // w_64^13
	u64(4398046511104ull),		  // w_64^14
	u64(35184372088832ull),		  // w_64^15
	u64(281474976710656ull),	  // w_64^16
	u64(2251799813685248ull),	  // w_64^17
	u64(18014398509481984ull),	  // w_64^18
	u64(144115188075855872ull),	  // w_64^19
	u64(1152921504606846976ull),  // w_64^20
	u64(9223372036854775808ull),  // w_64^21
	u64(17179869180ull),		  // w_64^22
	u64(137438953440ull),		  // w_64^23
	u64(1099511627520ull),		  // w_64^24
	u64(8796093020160ull),		  // w_64^25
	u64(70368744161280ull),		  // w_64^26
	u64(562949953290240ull),	  // w_64^27
	u64(4503599626321920ull),	  // w_64^28
	u64(36028797010575360ull),	  // w_64^29
	u64(288230376084602880ull),	  // w_64^30
	u64(2305843008676823040ull),  // w_64^31
	u64(18446744069414584320ull), // w_64^32
};

#endif