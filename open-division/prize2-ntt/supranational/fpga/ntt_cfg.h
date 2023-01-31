// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

#pragma once
namespace ntt_cfg
{
  const unsigned int NLANE = 16;

  unsigned int get_channel_from_point(unsigned long p) {

    unsigned int ch;

    ch  = (p >>  1) % (NLANE/2);
    ch += (p >> 13) % (NLANE/2);
    ch %= NLANE/2;

    ch *= 2;
    ch += ((p >> 23) ^ (p >> 11)) & 1;

    return ch;
  }

  unsigned int idx_to_addr(unsigned int idx) {
    return idx << 3;  // log2 size of unsigned long
  }

  unsigned int addr_to_idx(unsigned int addr) {
    return addr >> 3; // log2 size of unsigned long
  }

  unsigned int bank_optimize_addr(unsigned int addr) {

    unsigned int result;

    result   = ((addr >> 14) & ((1 <<  9)-1));
    result <<= 3;
    result  |= ((addr >> 15) & ((1 <<  3)-1)) ^
               ((addr >> 11) & ((1 <<  3)-1));
    result <<= 11;
    result  |= ((addr >>  0) & ((1 << 11)-1));

    return result;
  }

  unsigned int get_index_from_point(unsigned long p, unsigned int last) {

    unsigned int idx;
    unsigned int addr;

    if (last) {
      idx   = ((p >> 16) & ((1 <<  2)-1));// 17 16
      idx <<= 8;
      idx  |= ((p >>  4) & ((1 <<  8)-1));// 11 10 9 8 7 6 5 4
      idx <<= 5;
      idx  |= ((p >> 18) & ((1 <<  5)-1));// 22 21 20 19 18
      idx <<= 4;
      idx  |= ((p >>  0) & ((1 <<  4)-1));// 3 2 1 0
      idx <<= 1;
      idx  |= ((p >> 12) & ((1 <<  1)-1));// 12
    } else {
      idx   = ((p >> 16) & ((1 <<  2)-1));// 17 16
      idx <<= 8;
      idx  |= ((p >>  4) & ((1 <<  8)-1));// 11 10 9 8 7 6 5 4
      idx <<= 5;
      idx  |= ((p >> 18) & ((1 <<  5)-1));// 22 21 20 19 18
      idx <<= 4;
      idx  |= ((p >> 12) & ((1 <<  4)-1));// 15 14 13 12
      idx <<= 1;
      idx  |= ((p >>  0) & ((1 <<  1)-1));// 0
    };

    addr = idx_to_addr(idx);
    addr = bank_optimize_addr(addr);
    idx  = addr_to_idx(addr);

    return idx;
  }
}
