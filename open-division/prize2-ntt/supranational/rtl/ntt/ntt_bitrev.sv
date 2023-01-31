// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module ntt_bitrev
  import math_pkg::*;
  #(
    parameter N  = 2**11,  // number of pairs of points 
    parameter NLEVEL0 = 0,
    parameter NOP = 0
    )
  (
   input logic              rst_ni,
   input logic              clk_i,
   
   input logic [1:0][63:0]  x_i,
   input logic              valid_i,
  
   output logic [1:0][63:0] x_o,
   output logic             valid_o
   );
  
  logic [1:0][$clog2(N):0]  wa;
  logic [$clog2(N)+1:0]     wcnt_q;
  logic [1:0][63:0]         wx;
  logic                     rvalid, rvalid_p1_q, rvalid_p2_q;
  logic [1:0][$clog2(N):0]  ra;
  logic [$clog2(N)+1:0]     rcnt_q;
  logic [$clog2(N)-1:0]     wo, bitrev;
  logic [1:0][63:0]         rx;
  logic [1:0][63:0]         x;
  logic                     valid;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      wcnt_q <= 0;
    end
    else begin
      wcnt_q <= wcnt_q + valid_i;
    end
  end

  always_comb begin
    wa[0] = wcnt_q;
    wa[1] = wcnt_q;
  end

  always_comb begin
    wx[0] = x_i[0];
    wx[1] = x_i[1];
  end

  ram_1w1r_1clk
    #( 
       .ADDR_WIDTH ($clog2(N)+1),
       .DATA_WIDTH (64) 
       )
  ram_x0
    (
     .clk_i ( clk_i                              ),
     .a_a_i   ( wa[0]                              ),
     .a_wd_i  ( wx[0]                              ),
     .a_we_i  ( valid_i                  ),
     .b_a_i   ( ra[0]                              ),
     .b_re_i  ( rvalid || rvalid_p1_q ),
     .b_rd_o  ( rx[0]                              )
     );

  ram_1w1r_1clk
    #( 
       .ADDR_WIDTH ($clog2(N)+1),
       .DATA_WIDTH (64) 
       )
  ram_x1
    (
     .clk_i ( clk_i                              ),
     .a_a_i   ( wa[1]                              ),
     .a_wd_i  ( wx[1]                              ),
     .a_we_i  ( valid_i                  ),
     .b_a_i   ( ra[1]                              ),
     .b_re_i  ( rvalid || rvalid_p1_q ),
     .b_rd_o  ( rx[1]                              )
     );

  always_comb begin
    rvalid = wcnt_q[$clog2(N)+1:$clog2(N)] != rcnt_q[$clog2(N)+1:$clog2(N)];
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rcnt_q <= 0;
      rvalid_p1_q <= 0;
      rvalid_p2_q <= 0;
    end
    else begin
      rcnt_q <= rcnt_q + rvalid;
      rvalid_p1_q <= rvalid;
      rvalid_p2_q <= rvalid_p1_q;
    end
  end

  always_comb begin
    wo = rcnt_q[$clog2(N)-1:0];
    wo = {wo, wo} >> NLEVEL0;
    if (!NOP) begin
      for (int i=0; i<$clog2(N); i++) begin
        bitrev[i] = wo[$clog2(N)-1-i];
      end
    end
    else begin
        bitrev = wo;
    end
    ra[0] = {rcnt_q[$clog2(N)], bitrev};
    ra[1] = {rcnt_q[$clog2(N)], bitrev};
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      valid_o <= 0;
    end
    else begin
      valid_o <= rvalid_p2_q;
    end
  end

  always_ff @(posedge clk_i) begin
    x_o[0] <= rx[0];
    x_o[1] <= rx[1];
  end

`ifdef NEVER
  `ifndef SYNTHESIS

  string s;
  int    fdi, fdo, icnt, ocnt;

  initial begin
    $sformat(s,"%m_in.log");
    fdi = $fopen(s,"w");
    $sformat(s,"%m_out.log");
    fdo = $fopen(s,"w");
    icnt = 0;
    ocnt = 0;
  end

  always @(posedge clk_i) begin
    if (valid_i) begin
      $sformat(s,"%0d: ",icnt);
      $fwrite(fdi,s);
      for (int i=0; i<2; i++) begin
        $sformat(s,"[%0d] 0x%x ",i,x_i[i]);
        $fwrite(fdi,s);
      end
      $fwrite(fdi,"\n");
      icnt=icnt+1;
    end
    if (valid_o) begin
      $sformat(s,"%0d: ",ocnt);
      $fwrite(fdo,s);
      for (int i=0; i<2; i++) begin
        $sformat(s,"[%0d] 0x%x ",i,x_o[i]);
        $fwrite(fdo,s);
      end
      $fwrite(fdo,"\n");
      ocnt=ocnt+1;
    end
  end

  `endif
`endif

endmodule
