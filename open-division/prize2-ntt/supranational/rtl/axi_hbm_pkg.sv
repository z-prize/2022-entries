// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

package axi_hbm_pkg;

  localparam int DATA_WIDTH_IN_BYTES = 32;

  localparam int N_HBM_PC = 32;

  typedef logic  [32:0] addr_t;
  typedef logic   [5:0] wid_t;
  typedef logic   [5:0] rid_t;
  typedef logic   [3:0] len_t;

  localparam addr_t HBM_PC_SIZE_IN_BYTES = 1 << 28; // 256 MB

  typedef logic [DATA_WIDTH_IN_BYTES*8-1:0] data_t;
  typedef logic [DATA_WIDTH_IN_BYTES-1:0]   strb_t;

  function automatic strb_t parity(input data_t data);
    for (int i = 0; i < DATA_WIDTH_IN_BYTES; i++) begin
      parity[i] = ^data[i*8 +: 8];
    end
  endfunction

endpackage
