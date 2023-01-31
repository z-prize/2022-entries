// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module mulred
  import math_pkg::*;
  #(
    parameter BFLYDSP = 24  // 24, 16, or 12
    )
  (
   input logic 	       rst_ni,
   input logic 	       clk_i,
   input 	       ce_i,

   input logic [63:0]  a_i,
   input logic [63:0]  b_i,
   
   output logic [63:0] r_o
   );

  logic [127:0]        p;
  
  mul64x64 mul64x64
    ( .rst_ni,
      .clk_i,
      .ce_i,
      .a_i,
      .b_i,
      .p_o(p)
      );

  red128t64
      #( .NO_DSP_ADD32    ( BFLYDSP==24 ? 1 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_SUB64    ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 0 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_ADDSUB64 ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 0 : BFLYDSP==12 ? 1 : 0 )
         )
  red128t64
    ( .rst_ni,
      .clk_i,
      .ce_i,
      .p_i(p),
      .r_o
      );

`ifndef SYNTHESIS
  logic [PIPE_DEPTH_MULRED-1:0][63:0] a_q, b_q;
  logic [PIPE_DEPTH_MULRED-1:0] v_q;
  logic [127:0] pp;
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      v_q <= 0;
    end
    else if (glbl.GSR) begin
      v_q <= 0;
    end
    else if (ce_i) begin
      v_q <= {v_q, 1'b1};
      a_q <= {a_q, a_i};
      b_q <= {b_q, b_i};
    end
  end
  logic [65:0] rexp;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      pp = a_q[PIPE_DEPTH_MULRED-1] * b_q[PIPE_DEPTH_MULRED-1];
      rexp = pp[63:0] - pp[96+:32] + {pp[64+:32],32'h0} - pp[64+:32];
      if (rexp[65]) begin
        rexp += 1 << 64;
        rexp -= 32'hffffffff;
      end
      else if (rexp[64]) begin
        rexp -= 1 << 64;
        rexp += 32'hffffffff;
      end
      if (v_q[PIPE_DEPTH_MULRED-1] && rexp!=r_o) begin
        $display("v=%b a=%x b=%x ract=%x rexp=%x ERROR",
		 v_q[PIPE_DEPTH_MULRED-1],
		 a_q[PIPE_DEPTH_MULRED-1],b_q[PIPE_DEPTH_MULRED-1],r_o,rexp);
        $finish;
      end
      else begin
 `ifdef NEVER
        $display("v=%b a=%x b=%x ract=%x rexp=%x",
		 v_q[PIPE_DEPTH_MULRED-1],
		 a_q[PIPE_DEPTH_MULRED-1],b_q[PIPE_DEPTH_MULRED-1],r_o,rexp);
 `endif
      end
    end
  end
`endif

endmodule
