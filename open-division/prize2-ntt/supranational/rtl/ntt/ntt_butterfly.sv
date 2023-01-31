// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module ntt_butterfly
  import math_pkg::*;
  import ntt_opt_pkg::*;
  #(
    parameter NOP = 0,
    parameter LEVEL_FROM_END = 0,
    parameter BFLYDSP = 24  // 24, 16 or 12
    )
  (
   input logic              rst_ni,
   input logic              clk_i,
  
   input logic [1:0][63:0]  x_i,
   input logic [1:0][63:0]  w_i,
   input logic              valid_i,
  
   output logic [1:0][63:0] x_o,
   output logic [1:0][63:0] w_o,
   output logic             valid_o
   );

  localparam MODE
    = (LEVEL_FROM_END==1 && OPTIMIZE_FINAL_BFLY_STAGES) ? BUTTERFLY_W0    :
      (LEVEL_FROM_END==2 && OPTIMIZE_FINAL_BFLY_STAGES) ? BUTTERFLY_W0_W2 :
      BUTTERFLY_GENERIC;
  
  localparam PD 
    = (MODE == BUTTERFLY_W0   ) ? PIPE_DEPTH_BUTTERFLY_W0    :
      (MODE == BUTTERFLY_W0_W2) ? PIPE_DEPTH_BUTTERFLY_W0_W2 :
      PIPE_DEPTH_BUTTERFLY;
                 
  logic [PD-2:0][63:0] w;
  logic [PD-2:0]       valid_q;
  
  butterfly 
    #(
      .MODE(MODE), 
      .BFLYDSP(BFLYDSP),
      .CANONICAL((LEVEL_FROM_END==1) ? 1 : 0)
      ) 
  butterfly
     (
     .rst_ni( 1'b1              ),
     .clk_i,
     .nop_i ( NOP ? 1'b1 : 1'b0 ),
     .ce_i  ( 1'b1              ),
     .x_i   ( x_i[0]            ),
     .y_i   ( x_i[1]            ),
     .w_i   ( w_i[1]            ),
     .x_o   ( x_o[0]            ),
     .y_o   ( x_o[1]            )
     );
    
  always_ff @(posedge clk_i) begin
    {w_o[0], w} <= { w, w_i[0] };
  end

  if (OPTIMIZE_TWIDDLE_PIPELINING) begin
    // help tools to optimize away logic as ntt_cgram.sv will replicate
    assign w_o[1] = 0;
  end
  else begin
    assign w_o[1] = w_o[0];
  end
  
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      {valid_o, valid_q} <= '0;
    end
    else begin
      {valid_o, valid_q} <= {valid_q, valid_i};
    end
  end

endmodule
