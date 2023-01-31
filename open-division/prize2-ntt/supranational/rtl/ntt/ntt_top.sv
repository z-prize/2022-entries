// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module ntt_top
  #(
    parameter NLEVEL  = 12, // number of butterfly levels (2^12 NTT requires 12 levels)
    parameter NLEVEL0 =  5, // number of mini-cgram levels per stage
    parameter NLANE   =  8, // number of lanes to operate in parallel
    parameter PASS1   =  0, // 0 for pass 0 or 1 via pass1_i, 1 for pass 1 optimized
    parameter BFLYDSP = 12, // 24, 16, or 12
    parameter SLRX_X  =  1, // SLR0->SLR1 crossing on x_i, SLR1->SLR0 crossing on x_o
    parameter SLRX_W  =  1, // SLR0->SLR1 crossing on w
    parameter NOP     =  0  // useful for debugging, dataflow only no b-fly calcs
    )
  (
   input logic                         rst_ni,
   input logic                         clk_i,

   input logic                         pass1_i,

   input logic [NLANE-1:0][1:0][63:0]  x_i,
   input logic [NLANE-1:0]             valid_i,
   
   output logic [NLANE-1:0][1:0][63:0] x_o,
   output logic [NLANE-1:0]            valid_o
   );

   logic [NLANE-1:0][1:0][63:0]        w;
   logic [NLANE-1:0][1:0][63:0]        w_slrx, w_slr1;
   logic [NLANE-1:0][1:0][63:0]        x_slrx_i, x_slr1_i;
   logic [NLANE-1:0]                   valid_slrx_i, valid_slr1_i, ready_w;
   logic [NLANE-1:0][1:0][63:0]        x_slrx_o, x_slr1_o;
   logic [NLANE-1:0]		       valid_slrx_o, valid_slr1_o;

   (* dont_touch = "true" *) logic twid_rst_dt_qn;
   (* dont_touch = "true" *) logic core_rst_dt_qn;
   logic twid_rst_qn;
   logic core_rst_qn;
   always_ff @(posedge clk_i) begin
      twid_rst_dt_qn <= rst_ni;
      core_rst_dt_qn <= rst_ni;
      twid_rst_qn <= twid_rst_dt_qn;
      core_rst_qn <= core_rst_dt_qn;
   end
   
   ntt_twiddle
     #(
       .NLEVEL(NLEVEL),
       .NLEVEL0(NLEVEL0),
       .NLANE(NLANE),
       .PASS1_ONLY(PASS1),
       .BFLYDSP(BFLYDSP)
       )
   ntt_twiddle
     (
      .rst_ni(twid_rst_qn),
      .clk_i,
      .pass1_i(pass1_i),
      .ready_i(ready_w),
      .w_o(w)
      );

   if (SLRX_X) begin : slrx_xi
     for (genvar gv_i=0;gv_i<NLANE;gv_i++) begin : xi
       slrx_tx_reg #(.N(2*64+1))
       slrx_tx_reg_i
          ( .clk_i,
            .d_i    ( {     valid_i[gv_i],      x_i[gv_i]} ),
            .tx_reg ( {valid_slrx_i[gv_i], x_slrx_i[gv_i]} ) );
     
       slrx_rx_reg #(.N(2*64+1))
       slrx_rx_reg_i
          ( .clk_i,
            .d_i    ( {valid_slrx_i[gv_i], x_slrx_i[gv_i]} ),
            .rx_reg ( {valid_slr1_i[gv_i], x_slr1_i[gv_i]} ) );
     end
   end
   else begin
     assign x_slr1_i = x_i;
     assign valid_slr1_i = valid_i;
   end
   
   if (SLRX_W) begin : slrx_w
     for (genvar gv_i=0;gv_i<NLANE;gv_i++) begin : wi
       slrx_tx_reg #(.N(2*64))
       slrx_tx_reg_i
          ( .clk_i,
            .d_i    ( w[gv_i] ),
            .tx_reg ( w_slrx[gv_i] ) );
     
       slrx_rx_reg #(.N(2*64))
       slrx_rx_reg_i
          ( .clk_i,
            .d_i    ( w_slrx[gv_i] ),
            .rx_reg ( w_slr1[gv_i] ) );
     end
     assign ready_w = valid_i;
   end
   else begin
     assign w_slr1 = w;
     assign ready_w = valid_slr1_i;
   end
   
   ntt
     #(
       .NLEVEL(NLEVEL),
       .NLEVEL0(NLEVEL0),
       .NLANE(NLANE),
       .PASS(PASS1),
       .BFLYDSP(BFLYDSP)
       )
   ntt
     (
      .rst_ni(core_rst_qn),
      .clk_i,
      .x_i(x_slr1_i),
      .w_i(w_slr1),
      .valid_i(valid_slr1_i),
      .x_o(x_slr1_o),
      .valid_o(valid_slr1_o)
      );

   if (SLRX_X) begin : slrx_xo
     for (genvar gv_i=0;gv_i<NLANE;gv_i++) begin : xo
       slrx_tx_reg #(.N(2*64+1))
       slrx_tx_reg_o
          ( .clk_i,
            .d_i    ( {valid_slr1_o[gv_i], x_slr1_o[gv_i]} ),
            .tx_reg ( {valid_slrx_o[gv_i], x_slrx_o[gv_i]} ) );
     
       slrx_rx_reg #(.N(2*64+1))
       slrx_rx_reg_o
          ( .clk_i,
            .d_i    ( {valid_slrx_o[gv_i], x_slrx_o[gv_i]} ),
            .rx_reg ( {     valid_o[gv_i],      x_o[gv_i]} ) );
     end
   end
   else begin
     assign x_o = x_slr1_o;
     assign valid_o = valid_slr1_o;
   end


endmodule
