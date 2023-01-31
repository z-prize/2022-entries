// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

(* keep_hierarchy = "yes" *) module slrx_tx_reg
  #(
    parameter N = 5
    )
  (
   input logic         clk_i,
   input logic [N-1:0] d_i,
   (* dont_touch = "true" *) output logic [N-1:0] tx_reg
   );
  
  always_ff @(posedge clk_i) begin
    tx_reg <= d_i;
  end
  
endmodule

