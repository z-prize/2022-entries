// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

package math_pkg;

  localparam unsigned BUTTERFLY_GENERIC = 0;
  localparam unsigned BUTTERFLY_W0      = 1;
  localparam unsigned BUTTERFLY_W0_W2   = 2;

  localparam unsigned PIPE_DEPTH_MODADDSUB =  6;
  localparam unsigned PIPE_DEPTH_MUL64X64  = 14;
  localparam unsigned PIPE_DEPTH_RED128T64 =  8;
  localparam unsigned PIPE_DEPTH_MULRED    = PIPE_DEPTH_MUL64X64 + PIPE_DEPTH_RED128T64;
  localparam unsigned PIPE_DEPTH_BUTTERFLY = PIPE_DEPTH_MODADDSUB + PIPE_DEPTH_MULRED;

  localparam unsigned PIPE_DEPTH_BUTTERFLY_W0 = PIPE_DEPTH_MODADDSUB;

  localparam unsigned PIPE_DEPTH_BUTTERFLY_W0_W2 = PIPE_DEPTH_MODADDSUB + PIPE_DEPTH_RED128T64;

endpackage
