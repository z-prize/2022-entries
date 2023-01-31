// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module ntt_cgram
  import math_pkg::*;
  import ntt_opt_pkg::*;
  #(
    parameter N = 2**11,        // number of pairs of points 
    parameter LEVEL = 0,        // butterfly level which this cgram feeds
    parameter NLEVEL0 = 0,
    parameter NO_TWIDDLES = 0
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
  
  logic [1:0][$clog2(N):0]  wa;
  logic [$clog2(N)+1:0]     wcnt_q;
  logic                     wshift;
  logic [1:0][63:0]         wx, ww;
  logic                     rvalid, rvalid_p1_q, rvalid_p2_q;
  logic [1:0][$clog2(N):0]  ra;
  logic [$clog2(N)+1:0]     rcnt_q;
  logic [$clog2(N)-1:0]     wo;
  logic                     rshift, rshift_p1_q, rshift_p2_q;
  logic [1:0][63:0]         rx, rw;
  logic [1:0][63:0]         x, w;
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
    wshift = wcnt_q[$clog2(N)-1];
  end

  always_comb begin
    wx[0] = x_i[ wshift];
    wx[1] = x_i[!wshift];
    ww[0] = w_i[ wshift];
    ww[1] = w_i[!wshift];
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
     .a_we_i  ( valid_i                            ),
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

  if (NO_TWIDDLES==0) begin : twiddle
    if (OPTIMIZE_TWIDDLE_PIPELINING==0) begin : no_wopt

      ram_1w1r_1clk
        #( 
           .ADDR_WIDTH ($clog2(N)+1),
           .DATA_WIDTH (64) 
           )
      ram_w0
        (
         .clk_i ( clk_i                              ),
         .a_a_i   ( wa[0]                              ),
         .a_wd_i  ( ww[0]                              ),
         .a_we_i  ( valid_i                            ),
         .b_a_i   ( ra[0]                              ),
         .b_re_i  ( rvalid || rvalid_p1_q ),
         .b_rd_o  ( rw[0]                              )
         );
    
      ram_1w1r_1clk
        #( 
           .ADDR_WIDTH ($clog2(N)+1),
           .DATA_WIDTH (64) 
           )
      ram_w1
        (
         .clk_i ( clk_i                              ),
         .a_a_i   ( wa[1]                              ),
         .a_wd_i  ( ww[1]                              ),
         .a_we_i  ( valid_i                  ),
         .b_a_i   ( ra[1]                              ),
         .b_re_i  ( rvalid || rvalid_p1_q ),
         .b_rd_o  ( rw[1]                              )
         );
      
    end
    else begin : wopt
      
      localparam W_ADDR_WIDTH = $clog2(N)+1-LEVEL;
      logic [1:0][W_ADDR_WIDTH-1:0] _wa;
      logic [1:0][W_ADDR_WIDTH-1:0] _ra;
      logic [1:0]                   _we;
      logic [1:0]                   rsel_q;
      logic [1:0][63:0]             _rw;
      always_comb begin
        _wa[0] = (wa[0][$clog2(N)]<<($clog2(N)-LEVEL)) | ((wa[0]>>(LEVEL-1)) & ((1<<($clog2(N)-LEVEL))-1));
        _wa[1] = (wa[1][$clog2(N)]<<($clog2(N)-LEVEL)) | ((wa[1]>>(LEVEL-1)) & ((1<<($clog2(N)-LEVEL))-1));
        _ra[0] = (ra[0][$clog2(N)]<<($clog2(N)-LEVEL)) | ((ra[0]>>(LEVEL-1)) & ((1<<($clog2(N)-LEVEL))-1));
        _ra[1] = (ra[1][$clog2(N)]<<($clog2(N)-LEVEL)) | ((ra[1]>>(LEVEL-1)) & ((1<<($clog2(N)-LEVEL))-1));
        _we[0] = (wa[0][$clog2(N)-1]==0) && ( (wa[0] & ((1<<(LEVEL-1))-1)) == 0 );
        _we[1] = (wa[1][$clog2(N)-1]==1) && ( (wa[1] & ((1<<(LEVEL-1))-1)) == 0 );
      end
      always_ff @(posedge clk_i) begin
        rsel_q <= {rsel_q,ra[0][$clog2(N)-1]};
      end
  
      ram_1w1r_1clk
        #( 
           .ADDR_WIDTH (W_ADDR_WIDTH),
           .DATA_WIDTH (64) 
           )
      ram_w0
        (
         .clk_i ( clk_i                              ),
         .a_a_i   ( _wa[0]                              ),
         .a_wd_i  ( ww[0]                              ),
         .a_we_i  ( valid_i && _we[0]                  ),
         .b_a_i   ( _ra[0]                              ),
         .b_re_i  ( rvalid || rvalid_p1_q ),
         .b_rd_o  ( _rw[0]                              )
         );
  
      ram_1w1r_1clk
        #( 
           .ADDR_WIDTH (W_ADDR_WIDTH),
           .DATA_WIDTH (64) 
           )
      ram_w1
        (
         .clk_i ( clk_i                              ),
         .a_a_i   ( _wa[1]                              ),
         .a_wd_i  ( ww[1]                              ),
         .a_we_i  ( valid_i && _we[1]                  ),
         .b_a_i   ( _ra[1]                              ),
         .b_re_i  ( rvalid || rvalid_p1_q ),
         .b_rd_o  ( _rw[1]                              )
         );
  
      assign rw[0] = _rw[rsel_q[1]];
      assign rw[1] = _rw[!rsel_q[1]];
  
    end
  end

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
    rshift = wo[0];
    ra[0] = ({rcnt_q[$clog2(N)], wo[0]} << $clog2(N/2)) | ((wo>>1) & (N/2-1));
    ra[1] = ({rcnt_q[$clog2(N)],!wo[0]} << $clog2(N/2)) | ((wo>>1) & (N/2-1));
  end

  always_ff @(posedge clk_i) begin
    rshift_p1_q <= rshift;
    rshift_p2_q <= rshift_p1_q;
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
    x_o[0] <= rx[ rshift_p2_q];
    x_o[1] <= rx[!rshift_p2_q];
  end
  if (NO_TWIDDLES==0) begin
    always_ff @(posedge clk_i) begin
      w_o[0] <= rw[ rshift_p2_q];
      w_o[1] <= rw[!rshift_p2_q];
    end
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
        $sformat(s,"[%0d] x 0x%x w 0x%x ",i,x_i[i],w_i[i]);
        //$sformat(s,"[%0d] 0x%x ",i,x_i[i]);
        //$sformat(s,"[%0d] 0x%x ",i,w_i[i]);
        $fwrite(fdi,s);
      end
      $fwrite(fdi,"\n");
      icnt=icnt+1;
    end
    if (valid_o) begin
      $sformat(s,"%0d: ",ocnt);
      $fwrite(fdo,s);
      for (int i=0; i<2; i++) begin
        $sformat(s,"[%0d] x 0x%x w 0x%x ",i,x_o[i],w_o[i]);
        //$sformat(s,"[%0d] 0x%x ",i,x_o[i]);
        //$sformat(s,"[%0d] 0x%x ",i,w_o[i]);
        $fwrite(fdo,s);
      end
      $fwrite(fdo,"\n");
      ocnt=ocnt+1;
    end
  end

  `endif
`endif

endmodule
