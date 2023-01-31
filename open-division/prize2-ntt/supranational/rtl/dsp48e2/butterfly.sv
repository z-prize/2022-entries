// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module butterfly
  import math_pkg::*;
  #(
    parameter MODE = BUTTERFLY_GENERIC,
    parameter BFLYDSP = 24,  // 24, 16, or 12
    parameter CANONICAL = 0
    )
  (
   input logic         rst_ni,
   input logic         clk_i,
   input               ce_i,
   input               nop_i,

   input logic [63:0]  x_i,
   input logic [63:0]  y_i,
   input logic [63:0]  w_i,
   
   output logic [63:0] x_o,
   output logic [63:0] y_o
   );
  
  if (MODE == BUTTERFLY_W0) begin

    modaddsub
      #( .NO_DSP_ADD64    ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_SUB64    ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_ADDSUB64 ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .CANONICAL       ( CANONICAL )
         )
      modaddsub
      ( .rst_ni,
        .clk_i,
        .ce_i,
        .nop_i,
        .x_i,
        .y_i,
        .x_add_y_o(x_o),
        .x_sub_y_o(y_o)
        );
    
  end
  else if (MODE == BUTTERFLY_W0_W2) begin

    logic [63:0]         x_add_y;
    logic [63:0]         x_sub_y;
    modaddsub
      #( .NO_DSP_ADD64    ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_SUB64    ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_ADDSUB64 ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .CANONICAL       ( CANONICAL )
         )
    modaddsub
      ( .rst_ni,
        .clk_i,
        .ce_i,
        .nop_i,
        .x_i,
        .y_i,
        .x_add_y_o(x_add_y),
        .x_sub_y_o(x_sub_y)
        );
    
    logic [PIPE_DEPTH_MODADDSUB-1:0] w_q;
    always_ff @(posedge clk_i) begin
      if (ce_i) begin
        w_q <= {w_q, nop_i ? 1'h1 : w_i[0]};
      end
    end

    logic [127:0]        p;
    assign p = w_q[PIPE_DEPTH_MODADDSUB-1] ? x_sub_y : (x_sub_y << 48);

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
        .r_o(y_o)
        );

    logic [PIPE_DEPTH_RED128T64-1:0][63:0]    x_add_y_q;
    always_ff @(posedge clk_i) begin
      if (ce_i) begin
        x_add_y_q <= {x_add_y_q, x_add_y};
      end
    end

    assign x_o = x_add_y_q[PIPE_DEPTH_RED128T64-1];

  end

  else /* if (MODE == BUTTERFLY_GENERIC) */ begin
  
    logic [63:0]         x_add_y;
    logic [63:0]         x_sub_y;
    modaddsub
      #( .NO_DSP_ADD64    ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_SUB64    ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .NO_DSP_ADDSUB64 ( BFLYDSP==24 ? 0 : BFLYDSP==16 ? 1 : BFLYDSP==12 ? 1 : 0 ),
         .CANONICAL       ( CANONICAL )
         )
    modaddsub
      ( .rst_ni,
        .clk_i,
        .ce_i,
        .nop_i,
        .x_i,
        .y_i,
        .x_add_y_o(x_add_y),
        .x_sub_y_o(x_sub_y)
        );
    
    logic [PIPE_DEPTH_MODADDSUB-1:0][63:0] w_q;
    always_ff @(posedge clk_i) begin
      if (ce_i) begin
        w_q <= {w_q, nop_i ? 64'h1 : w_i};
      end
    end

    mulred #(.BFLYDSP(BFLYDSP)) mulred
      ( .rst_ni,
        .clk_i,
        .ce_i,
        .a_i(x_sub_y),
        .b_i(w_q[PIPE_DEPTH_MODADDSUB-1]),
        .r_o(y_o)
        );

    logic [PIPE_DEPTH_MULRED-1:0][63:0]    x_add_y_q;
    always_ff @(posedge clk_i) begin
      if (ce_i) begin
        x_add_y_q <= {x_add_y_q, x_add_y};
      end
    end

    assign x_o = x_add_y_q[PIPE_DEPTH_MULRED-1];

  end

`ifndef SYNTHESIS

  localparam PD 
    = (MODE == BUTTERFLY_W0   ) ? PIPE_DEPTH_BUTTERFLY_W0    :
      (MODE == BUTTERFLY_W0_W2) ? PIPE_DEPTH_BUTTERFLY_W0_W2 :
      PIPE_DEPTH_BUTTERFLY;
                 
  logic [PD-1:0][63:0] x_i_q, y_i_q, w_i_q;
  logic [PD-1:0] nop_i_q;
  logic [PD-1:0] v_q;
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      v_q <= 0;
    end
    else if (ce_i) begin
      v_q <= {v_q, 1'b1};
      x_i_q <= {x_i_q, x_i};
      y_i_q <= {y_i_q, y_i};
      w_i_q <= {w_i_q, w_i};
      nop_i_q <= {nop_i_q, nop_i};
    end
  end
  logic [64:0] addexp, subexp;
  logic [127:0] pp;
  logic [65:0] rexp;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      addexp = nop_i_q[PD-1] ? x_i_q[PD-1] :
               (x_i_q[PD-1] + y_i_q[PD-1]);
      if (addexp[64]) begin
        addexp -= 1 << 64;
        addexp += 32'hffffffff;
      end
      if (addexp[64]) begin
        addexp -= 1 << 64;
        addexp += 32'hffffffff;
      end
      subexp = nop_i_q[PD-1] ? y_i_q[PD-1] :
               (x_i_q[PD-1] - y_i_q[PD-1]);
      if (subexp[64]) begin
        subexp += 1 << 64;
        subexp -= 32'hffffffff;
      end
      if (subexp[64]) begin
        subexp += 1 << 64;
        subexp -= 32'hffffffff;
      end
      pp = subexp * (nop_i_q[PD-1] ? 1 : w_i_q[PD-1]);
      rexp = pp[63:0] - pp[96+:32] + {pp[64+:32],32'h0} - pp[64+:32];
      if (rexp[65]) begin
        rexp += 1 << 64;
        rexp -= 32'hffffffff;
      end
      else if (rexp[64]) begin
        rexp -= 1 << 64;
        rexp += 32'hffffffff;
      end
      if (v_q[PD-1] && (addexp!=x_o || rexp!=y_o)) begin
        $display("v=%b x_i=%x y_i=%x w_i=%x nop_i=%d x_o=%x y_o=%x x_o_exp=%x y_o_exp=%x ERROR",
                 v_q[PD-1],
		 x_i_q[PD-1],y_i_q[PD-1],
		 w_i_q[PD-1],nop_i_q[PD-1],
                 x_o,y_o,addexp,rexp);
        $finish;
      end
      else begin
 `ifdef NEVER
        $display("v=%b x_i=%x y_i=%x w_i=%x nop_i=%d x_o=%x y_o=%x x_o_exp=%x y_o_exp=%x",
                 v_q[PD-1],
		 x_i_q[PD-1],y_i_q[PD-1],
		 w_i_q[PD-1],nop_i_q[PD-1],
                 x_o,y_o,addexp,rexp);
 `endif
      end
    end
  end
`endif

endmodule
