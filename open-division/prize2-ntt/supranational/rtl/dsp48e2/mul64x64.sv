// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module mul64x64
  import math_pkg::*;
  (
   input logic 		rst_ni,
   input logic 		clk_i,
   input 		ce_i,

   input logic [63:0] 	a_i,
   input logic [63:0] 	b_i,
   
   output logic [127:0] p_o
   );
   
  // pipe along inputs ... many of these flops will be optimized away
  logic [PIPE_DEPTH_MUL64X64:0][63:0] a, b;
  logic [PIPE_DEPTH_MUL64X64-1:0][63:0] a_q, b_q;
  always_comb begin
    a = {a_q, a_i};
    b = {b_q, b_i};
  end
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      a_q <= a;
      b_q <= b;
    end
  end

  
  logic [2:0][3:0][47:0] dsp_p, dsp_p_q, dsp_pc;
  logic [2:0][3:0][29:0] dsp_ac;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      dsp_p_q <= dsp_p;
    end
  end

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (0),
       .A_CASCADE        (0),
       .Z_SEL_17BITSHIFT (0),
       .W_SEL_C          (0) )
  dsp_0_0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i({4'b0,a[0][0*26 +:26]}),
      .ACIN_i(30'h0),
      .B_i({1'b0,b[0][0*17 +:17]}),
      .C_i(),
      .PCIN_i(),
      .ACOUT_o(dsp_ac[0][0]),
      .P_o(dsp_p[0][0]),
      .PCOUT_o(dsp_pc[0][0])
      );

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (0) )
  dsp_0_1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[0][0]),
      .B_i({1'b0,b[0][1*17 +:17]}),
      .C_i(),
      .PCIN_i(dsp_pc[0][0]),
      .ACOUT_o(dsp_ac[0][1]),
      .P_o(dsp_p[0][1]),
      .PCOUT_o(dsp_pc[0][1])
      );

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (0) )
  dsp_0_2
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[0][1]),
      .B_i({1'b0,b[1][2*17 +:17]}),
      .C_i(),
      .PCIN_i(dsp_pc[0][1]),
      .ACOUT_o(dsp_ac[0][2]),
      .P_o(dsp_p[0][2]),
      .PCOUT_o(dsp_pc[0][2])
      );

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (0) )
  dsp_0_3
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[0][2]),
      .B_i({1'b0,4'b0,b[2][63: 3*17]}),
      .C_i(),
      .PCIN_i(dsp_pc[0][2]),
      .ACOUT_o(dsp_ac[0][3]),
      .P_o(dsp_p[0][3]),
      .PCOUT_o(dsp_pc[0][3])
      );



  dsp_mul64x64
    #( .A_DUAL_REG       (1),
       .B_DUAL_REG       (1),
       .A_CASCADE        (0),
       .Z_SEL_17BITSHIFT (0),
       .W_SEL_C          (1) )
  dsp_1_0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i({4'b0,a[3][1*26 +:26]}),
      .ACIN_i(30'h0),
      .B_i({1'b0,b[3][0*17 +:17]}),
      .C_i({{(48-17){1'b0}},dsp_p[0][2][(26-17-1):0],dsp_p_q[0][1][16:(26-17)]}),
      .PCIN_i(),
      .ACOUT_o(dsp_ac[1][0]),
      .P_o(dsp_p[1][0]),
      .PCOUT_o(dsp_pc[1][0])
      );

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (1) )
  dsp_1_1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[1][0]),
      .B_i({1'b0,b[4][1*17 +:17]}),
      .C_i({{(48-17){1'b0}},dsp_p[0][3][(26-17-1):0],dsp_p_q[0][2][16:(26-17)]}),
      .PCIN_i(dsp_pc[1][0]),
      .ACOUT_o(dsp_ac[1][1]),
      .P_o(dsp_p[1][1]),
      .PCOUT_o(dsp_pc[1][1])
      );

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (1) )
  dsp_1_2
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[1][1]),
      .B_i({1'b0,b[5][2*17 +:17]}),
      .C_i({{(48-39){1'b0}},dsp_p_q[0][3][47:(26-17)]}),
      .PCIN_i(dsp_pc[1][1]),
      .ACOUT_o(dsp_ac[1][2]),
      .P_o(dsp_p[1][2]),
      .PCOUT_o(dsp_pc[1][2])
      );
  
  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (0) )
  dsp_1_3
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[1][2]),
      .B_i({1'b0,4'b0,b[6][63: 3*17]}),
      .C_i(),
      .PCIN_i(dsp_pc[1][2]),
      .ACOUT_o(dsp_ac[1][3]),
      .P_o(dsp_p[1][3]),
      .PCOUT_o(dsp_pc[1][3])
      );



  dsp_mul64x64
    #( .A_DUAL_REG       (1),
       .B_DUAL_REG       (1),
       .A_CASCADE        (0),
       .Z_SEL_17BITSHIFT (0),
       .W_SEL_C          (1) )
  dsp_2_0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i({4'b0,{(30-16){1'b0}},a[7][63: 2*26]}),
      .ACIN_i(30'h0),
      .B_i({1'b0,b[7][0*17 +:17]}),
      .C_i({{(48-17){1'b0}},dsp_p[1][2][(26-17-1):0],dsp_p_q[1][1][16:(26-17)]}),
      .PCIN_i(),
      .ACOUT_o(dsp_ac[2][0]),
      .P_o(dsp_p[2][0]),
      .PCOUT_o(dsp_pc[2][0])
      );

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (1) )
  dsp_2_1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[2][0]),
      .B_i({1'b0,b[8][1*17 +:17]}),
      .C_i({{(48-17){1'b0}},dsp_p[1][3][(26-17-1):0],dsp_p_q[1][2][16:(26-17)]}),
      .PCIN_i(dsp_pc[2][0]),
      .ACOUT_o(dsp_ac[2][1]),
      .P_o(dsp_p[2][1]),
      .PCOUT_o(dsp_pc[2][1])
      );

 dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (1) )
  dsp_2_2
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[2][1]),
      .B_i({1'b0,b[9][2*17 +:17]}),
      .C_i({{(48-39){1'b0}},dsp_p_q[1][3][47:(26-17)]}),
      .PCIN_i(dsp_pc[2][1]),
      .ACOUT_o(dsp_ac[2][2]),
      .P_o(dsp_p[2][2]),
      .PCOUT_o(dsp_pc[2][2])
      );

  dsp_mul64x64
    #( .A_DUAL_REG       (0),
       .B_DUAL_REG       (1),
       .A_CASCADE        (1),
       .Z_SEL_17BITSHIFT (1),
       .W_SEL_C          (0) )
  dsp_2_3
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .A_i(30'h0),
      .ACIN_i(dsp_ac[2][2]),
      .B_i({1'b0,4'b0,b[10][63: 3*17]}),
      .C_i(),
      .PCIN_i(dsp_pc[2][2]),
      .ACOUT_o(dsp_ac[2][3]),
      .P_o(dsp_p[2][3]),
      .PCOUT_o(dsp_pc[2][3])
      );


  logic [14:0][127:0]    p, p_q;
  always_comb begin
    p[0] = 0;
    p[1] = p_q[0];
    p[2] = p_q[1];
    p[3] = p_q[2]     | (dsp_p[0][0][16:0]        <<    0);
    p[4] = p_q[3]     | (dsp_p[0][1][(26-17-1):0] <<   17);
    p[5] = p_q[4];
    p[6] = p_q[5];
    p[7] = p_q[6]     | (dsp_p[1][0][16:0]        <<   26);
    p[8] = p_q[7]     | (dsp_p[1][1][(26-17-1):0] <<   43);
    p[9] = p_q[8];
    p[10] = p_q[9];
    p[11] = p_q[10]   | (dsp_p[2][0][16:0]        <<   52);
    p[12] = p_q[11]   | (dsp_p[2][1][16:0]        <<   69);
    p[13] = p_q[12]   | (dsp_p[2][2][16:0]        <<   86);
    p[14] = p_q[13]   | (dsp_p[2][3]              <<  103);
    p_o = p[14];
  end
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      p_q <= p;
    end
  end

`ifndef SYNTHESIS
  logic [PIPE_DEPTH_MUL64X64-1:0] v_q;
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      v_q <= 0;
    end
    else if (glbl.GSR) begin
      v_q <= 0;
    end
    else if (ce_i) begin
      v_q <= {v_q, 1'b1};
    end
  end
  logic [127:0] pexp;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      pexp = a[PIPE_DEPTH_MUL64X64]*b[PIPE_DEPTH_MUL64X64];
      if (v_q[PIPE_DEPTH_MUL64X64-1] && pexp!=p_o) begin
        $display("v=%b a=%x b=%x pact=%x pexp=%x ERROR",
		 v_q[PIPE_DEPTH_MUL64X64-1],
		 a[PIPE_DEPTH_MUL64X64],b[PIPE_DEPTH_MUL64X64],p_o,pexp);
        $stop;
      end
      else begin
 `ifdef NEVER        
	 $display("v=%b a=%x b=%x pact=%x pexp=%x",
		  v_q[PIPE_DEPTH_MUL64X64-1],
		  a[PIPE_DEPTH_MUL64X64],b[PIPE_DEPTH_MUL64X64],p_o,pexp);
 `endif
      end
    end
  end
`endif

endmodule


//`define USE_DSP_ABSTRACT

module dsp_mul64x64
  #(
    parameter A_DUAL_REG       = 0,
    parameter B_DUAL_REG       = 0,
    parameter A_CASCADE        = 0, 
    parameter Z_SEL_17BITSHIFT = 0,
    parameter W_SEL_C          = 0
   )   
  (
   input logic         rst_ni,
   input logic         clk_i,
   input logic         ce_i,
   input logic [29:0]  A_i,
   input logic [29:0]  ACIN_i,
   input logic [17:0]  B_i,
   input logic [47:0]  C_i,
   input logic [47:0]  PCIN_i,
   output logic [29:0] ACOUT_o,
   output logic [47:0] P_o,
   output logic [47:0] PCOUT_o
   );

`ifdef USE_DSP_ABSTRACT

  logic signed [26:0]  a0_q;
  logic signed [17:0]  b0_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      a0_q <= A_CASCADE ? ACIN_i : A_i;
      b0_q <= B_i;
    end
  end

  logic signed [26:0]  a_q;
  logic signed [17:0]  b_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      a_q <= A_DUAL_REG ? a0_q : A_CASCADE ? ACIN_i : A_i;
      b_q <= B_DUAL_REG ? b0_q : B_i;
    end
  end

  logic signed [44:0]  c_q, m_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      m_q <= a_q * b_q;
      c_q <= C_i;
    end
  end

  logic signed [47:0] p_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      p_q <= m_q + (Z_SEL_17BITSHIFT ? (PCIN_i>>17) : 0) + (W_SEL_C ? c_q : 0);
    end
  end

`endif

  logic [3:0] alumode;
  logic [4:0] inmode;
  logic [8:0] opmode;
  logic [47:0] dsp48e2_p, dsp48e2_pcout;
  logic [29:0] dsp48e2_acout;

  always_comb begin
    alumode = 4'b0000;
    inmode = (A_DUAL_REG ? 5'b00000 : 5'b00001) | (B_DUAL_REG ? 5'b00000 : 5'b10000);
    opmode = {
              W_SEL_C ? 2'b11 : 2'b00,              // W <- C : 0
              Z_SEL_17BITSHIFT ? 3'b101 : 3'b000,   // Z <- PCIN>>17 : 0
              4'b0101                               // XY <- M
              };
  end

  DSP48E2 
    #(
      .ACASCREG(A_DUAL_REG ? 2 : 1),
      .ADREG(),
      .ALUMODEREG(),
      .AMULTSEL(),
      .AREG(A_DUAL_REG ? 2 : 1),
      .AUTORESET_PATDET(),
      .AUTORESET_PRIORITY(),
      .A_INPUT(A_CASCADE ? "CASCADE" : "DIRECT"),
      .BCASCREG(),
      .BMULTSEL(),
      .BREG(B_DUAL_REG ? 2 : 1),
      .B_INPUT(),
      .CARRYINREG(),
      .CARRYINSELREG(),
      .CREG(),
      .DREG(),
      .INMODEREG(),
      .IS_ALUMODE_INVERTED(),
      .IS_CARRYIN_INVERTED(),
      .IS_CLK_INVERTED(),
      .IS_INMODE_INVERTED(),
      .IS_OPMODE_INVERTED(),
      .IS_RSTALLCARRYIN_INVERTED(),
      .IS_RSTALUMODE_INVERTED(),
      .IS_RSTA_INVERTED(),
      .IS_RSTB_INVERTED(),
      .IS_RSTCTRL_INVERTED(),
      .IS_RSTC_INVERTED(),
      .IS_RSTD_INVERTED(),
      .IS_RSTINMODE_INVERTED(),
      .IS_RSTM_INVERTED(),
      .IS_RSTP_INVERTED(),
      .MASK(),
      .MREG(),
      .OPMODEREG(),
      .PATTERN(),
      .PREADDINSEL(),
      .PREG(),
      .RND(),
      .SEL_MASK(),
      .SEL_PATTERN(),
      .USE_MULT(),
      .USE_PATTERN_DETECT(),
      .USE_SIMD(),
      .USE_WIDEXOR(),
      .XORSIMD()
      )
  dsp
    (
     .ACOUT(dsp48e2_acout),
     .BCOUT(),
     .CARRYCASCOUT(),
     .CARRYOUT(),
     .MULTSIGNOUT(),
     .OVERFLOW(),
     .P(dsp48e2_p),
     .PATTERNBDETECT(),
     .PATTERNDETECT(),
     .PCOUT(dsp48e2_pcout),
     .UNDERFLOW(),
     .XOROUT(),

     .A(A_i),
     .ACIN(ACIN_i),
     .ALUMODE(alumode[3:0]),
     .B(B_i),
     .BCIN(18'h0),
     .C(C_i),
     .CARRYCASCIN(1'b0),
     .CARRYIN(1'b0),
     .CARRYINSEL(3'h0),
     .CEA1(ce_i),
     .CEA2(ce_i),
     .CEAD(ce_i),
     .CEALUMODE(ce_i),
     .CEB1(ce_i),
     .CEB2(ce_i),
     .CEC(ce_i),
     .CECARRYIN(ce_i),
     .CECTRL(ce_i),
     .CED(ce_i),
     .CEINMODE(ce_i),
     .CEM(ce_i),
     .CEP(ce_i),
     .CLK(clk_i),
     .D(27'h0),
     .INMODE(inmode[4:0]),
     .MULTSIGNIN(1'b0),
     .OPMODE(opmode[8:0]),
     .PCIN(PCIN_i),
     .RSTA(!rst_ni),
     .RSTALLCARRYIN(!rst_ni),
     .RSTALUMODE(!rst_ni),
     .RSTB(!rst_ni),
     .RSTC(!rst_ni),
     .RSTCTRL(!rst_ni),
     .RSTD(!rst_ni),
     .RSTINMODE(!rst_ni),
     .RSTM(!rst_ni),
     .RSTP(!rst_ni)
     );

`ifdef USE_DSP_ABSTRACT
  wire match = (dsp48e2_p == p_q) && (dsp48e2_pcout == p_q) && (dsp48e2_acout == a_q);
  `undef USE_DSP_ABSTACT
`endif
  
  assign ACOUT_o = dsp48e2_acout;
  assign P_o = dsp48e2_p;
  assign PCOUT_o = dsp48e2_pcout;
  
endmodule
