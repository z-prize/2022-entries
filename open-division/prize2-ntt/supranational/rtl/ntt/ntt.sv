// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

(* keep_hierarchy = "yes" *) module ntt
  import ntt_opt_pkg::*;
  #(
    parameter NLEVEL  = 12, // number of butterfly levels (2^12 NTT requires 12 levels)
    parameter NLEVEL0 =  0, // number of mini-cgram levels per stage
    parameter NLANE   =  1, // number of lanes to operate in parallel
    parameter PASS    =  0, // 0 or 1
    parameter BFLYDSP = 24, // 24, 16, or 12
    parameter NOP     =  0  // useful for debugging, dataflow only no b-fly calcs
    )
  (
   input logic                         rst_ni,
   input logic                         clk_i,

   input logic [NLANE-1:0][1:0][63:0]  x_i,
   input logic [NLANE-1:0][1:0][63:0]  w_i,
   input logic [NLANE-1:0]             valid_i,
   
   output logic [NLANE-1:0][1:0][63:0] x_o,
   output logic [NLANE-1:0]            valid_o
   );

  localparam NLEVEL_PER_STAGE = NLEVEL0 + 1;
  localparam NSTAGE = NLEVEL / NLEVEL_PER_STAGE;

  initial assert(NSTAGE * NLEVEL_PER_STAGE == NLEVEL);
  
  wire [2*NLEVEL-1:0][NLANE-1:0][1:0][63:0]    x;
  wire [2*NLEVEL-1:0][NLANE-1:0][1:0][63:0]    w;
  wire [2*NLEVEL-1:0][NLANE-1:0]               valid;

  assign x[0] = x_i;
  assign w[0] = (PASS==0 || OPTIMIZE_PASS1_TWIDDLES==0) ? w_i : {NLANE{w_i[0]}};
  assign valid[0] = valid_i;

  for (genvar i=0; i<NLEVEL; i++) begin : level
    for (genvar j=0; j<NLANE; j++) begin : lane

      localparam k = i % NLEVEL_PER_STAGE;
    
      ntt_butterfly
        #(
          .NOP(NOP),
          .LEVEL_FROM_END((PASS==0 ? 2 : 1)*NLEVEL-i),
          .BFLYDSP(BFLYDSP)
          )
      ntt_butterfly
        (
         .rst_ni   ( rst_ni           ),
         .clk_i    ( clk_i            ),
         .x_i      ( x[2*i][j]        ),
         .w_i      ( w[2*i][j]        ),
         .valid_i  ( valid[2*i][j]    ),
         .x_o      ( x[2*i+1][j]      ),
         .w_o      ( w[2*i+1][j]      ),
         .valid_o  ( valid[2*i+1][j]  )
         );

      if (i<(NLEVEL-1)) begin
        if (PASS==0 || ((j%2)==0) || OPTIMIZE_PASS1_TWIDDLES==0) begin
          ntt_cgram
            #(
              .N           ( 1 << ((k<NLEVEL0) ? NLEVEL0 : (NLEVEL-1)) ),
              .LEVEL       ( (k<NLEVEL0) ? (k+1) : (i+1)               ),
              .NLEVEL0     ( (k<NLEVEL0) ? 0     : NLEVEL0             )
              )
          ntt_cgram
            (
             .rst_ni   ( rst_ni          ),
             .clk_i    ( clk_i           ),
             .x_i      ( x[2*i+1][j]     ),
             .w_i      ( w[2*i+1][j]     ),
             .valid_i  ( valid[2*i+1][j] ),
             .x_o      ( x[2*i+2][j]     ),
             .w_o      ( w[2*i+2][j]     ),
             .valid_o  ( valid[2*i+2][j] )
             );
        end
        else begin
          ntt_cgram
            #(
              .N           ( 1 << ((k<NLEVEL0) ? NLEVEL0 : (NLEVEL-1)) ),
              .LEVEL       ( (k<NLEVEL0) ? (k+1) : (i+1)               ),
              .NLEVEL0     ( (k<NLEVEL0) ? 0     : NLEVEL0             ),
              .NO_TWIDDLES ( 1                                         )
              )
          ntt_cgram
            (
             .rst_ni   ( rst_ni          ),
             .clk_i    ( clk_i           ),
             .x_i      ( x[2*i+1][j]     ),
             .w_i      ( w[2*i+1][j]     ),
             .valid_i  ( valid[2*i+1][j] ),
             .x_o      ( x[2*i+2][j]     ),
             .w_o      (                 ),
             .valid_o  ( valid[2*i+2][j] )
             );
          assign w[2*i+2][j] = w[2*i+2][0];
        end
      end
      else begin
        ntt_bitrev
          #(
            .N       ( 1 << (NLEVEL-1)  ),
            .NLEVEL0 ( NLEVEL0          )
            )
        ntt_bitrev
          (
           .rst_ni   ( rst_ni          ),
           .clk_i    ( clk_i           ),
           .x_i      ( x[2*i+1][j]     ),
           .valid_i  ( valid[2*i+1][j] ),
           .x_o      ( x_o[j]          ),
           .valid_o  ( valid_o[j]      )
           );
      end

    end
  end

`ifdef NEVER
  `ifndef SYNTHESIS

  for (genvar j=0; j<NLANE; j++) begin : lane

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
          $sformat(s,"[%0d] x 0x%x w 0x%x ",i,x_i[j][i],w_i[j][i]);
          //$sformat(s,"[%0d] 0x%x ",i,x_i[j][i]);
          //$sformat(s,"[%0d] 0x%x ",i,w_i[j][i]);
          $fwrite(fdi,s);
        end
        $fwrite(fdi,"\n");
        icnt=icnt+1;
      end
      if (valid_o) begin
        $sformat(s,"%0d: ",ocnt);
        $fwrite(fdo,s);
        for (int i=0; i<2; i++) begin
          //$sformat(s,"[%0d] x 0x%x w 0x%x ",i,x_o[j][i],w_o[j][i]);
          $sformat(s,"[%0d] 0x%x ",i,x_o[j][i]);
          //$sformat(s,"[%0d] 0x%x ",i,w_o[j][i]);
          $fwrite(fdo,s);
        end
        $fwrite(fdo,"\n");
        ocnt=ocnt+1;
      end
    end

  end

  `endif
`endif

endmodule
