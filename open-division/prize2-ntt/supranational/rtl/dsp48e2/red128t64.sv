// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module red128t64
  import math_pkg::*;
  #(
    parameter NO_DSP_ADD32    = 1,
    parameter NO_DSP_SUB64    = 0,
    parameter NO_DSP_ADDSUB64 = 0
    )
  (
   input logic         rst_ni,
   input logic         clk_i,
   input               ce_i,

   input logic [127:0] p_i,

   output logic [63:0] r_o
   );
  
  logic [47:0] dword3_p_dword2;
  dsp_red128t64
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (0),
       .USE_CARRY        (0),
       .NO_DSP           (NO_DSP_ADD32) )
  dsp_dword3_p_dword2
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({16'h0,p_i[96+:32]}),
      .C_i({16'h0,p_i[64+:32]}),
      .sub_i(1'b0),
      .nop_i(1'b0),
      .CARRYCASCIN_i(),
      .P_o(dword3_p_dword2),
      .CARRYCASCOUT_o()
      );

  logic [47:0] dword2_p_dword1;
  dsp_red128t64
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (0),
       .USE_CARRY        (0),
       .NO_DSP           (NO_DSP_ADD32) )
  dsp_dword2_p_dword1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({16'h0,p_i[64+:32]}),
      .C_i({16'h0,p_i[32+:32]}),
      .sub_i(1'b0),
      .nop_i(1'b0),
      .CARRYCASCIN_i(),
      .P_o(dword2_p_dword1),
      .CARRYCASCOUT_o()
      );

  logic [31:0] dword0_p1_q;
  logic [32:0] dword3_p_dword2_p1_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      dword0_p1_q <= p_i[0 +: 32];
      dword3_p_dword2_p1_q <= dword3_p_dword2;
    end
  end


  logic [47:0] r0_pre;
  logic        r0_pre_carry;
  dsp_red128t64
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (0),
       .NO_DSP           (NO_DSP_SUB64) )
  dsp_sub0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({dword3_p_dword2[31:0],16'h0}),
      .C_i({dword0_p1_q,16'h0}),
      .sub_i(1'b1),
      .nop_i(1'b0),
      .CARRYCASCIN_i(),
      .P_o(r0_pre),
      .CARRYCASCOUT_o(r0_pre_carry)
      );

  logic [47:0] r1_pre;
  dsp_red128t64
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (1),
       .NO_DSP           (NO_DSP_SUB64) )
  dsp_sub1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({47'h0,dword3_p_dword2_p1_q[32]}),
      .C_i({15'h0,dword2_p_dword1[32:0]}),
      .sub_i(1'b1),
      .nop_i(1'b0),
      .CARRYCASCIN_i(r0_pre_carry),
      .P_o(r1_pre),
      .CARRYCASCOUT_o()
      );

  logic        mod_nop, mod_nop_q;
  logic        mod_sub, mod_sub_q;

  always_comb begin
    mod_nop = r1_pre[33:32]==0;
    mod_sub = r1_pre[33];
  end
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      mod_nop_q <= mod_nop;
      mod_sub_q <= mod_sub;
    end
  end

  logic [47:0] r0;
  logic        r0_carry;
  dsp_red128t64
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (0),
       .NO_DSP           (NO_DSP_ADDSUB64) )
  dsp_mod0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({32'hffffffff,16'h0}),
      .C_i({r0_pre[47:16],16'h0}),
      .sub_i(mod_sub),
      .nop_i(mod_nop),
      .CARRYCASCIN_i(),
      .P_o(r0),
      .CARRYCASCOUT_o(r0_carry)
      );

  logic [47:0] r1;
  dsp_red128t64
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (1),
       .NO_DSP           (NO_DSP_ADDSUB64) )
  dsp_mod1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i(48'h0),
      .C_i({16'h0,r1_pre[31:0]}),
      .sub_i(mod_sub_q),
      .nop_i(mod_nop_q),
      .CARRYCASCIN_i(r0_carry),
      .P_o(r1),
      .CARRYCASCOUT_o()
      );

  logic [47:0] r0_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      r0_q <= r0;
    end
  end

  always_comb begin
    r_o = {r1[31:0],r0_q[47:16]};
  end

`ifndef SYNTHESIS
  logic [PIPE_DEPTH_RED128T64-1:0][127:0] p_q;
  logic [PIPE_DEPTH_RED128T64-1:0] v_q;
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      v_q <= 0;
    end
    else if (ce_i) begin
      v_q <= {v_q, 1'b1};
      p_q <= {p_q, p_i};
    end
  end
  logic [65:0] rexp;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      rexp = p_q[PIPE_DEPTH_RED128T64-1][63:0] 
	     - p_q[PIPE_DEPTH_RED128T64-1][96+:32] 
	     + {p_q[PIPE_DEPTH_RED128T64-1][64+:32],32'h0} 
	     - p_q[PIPE_DEPTH_RED128T64-1][64+:32];
      if (rexp[65]) begin
        rexp += 1 << 64;
        rexp -= 32'hffffffff;
      end
      else if (rexp[64]) begin
        rexp -= 1 << 64;
        rexp += 32'hffffffff;
      end
      if (v_q[PIPE_DEPTH_RED128T64-1] && rexp!=r_o) begin
        $display("v=%b p=%x ract=%x rexp=%x ERROR",
		 v_q[PIPE_DEPTH_RED128T64-1],
		 p_q[PIPE_DEPTH_RED128T64-1],r_o,rexp);
        $stop;
      end
      else begin
 `ifdef NEVER
         $display("v=%b p=%x ract=%x rexp=%x",
		  v_q[PIPE_DEPTH_RED128T64-1],
		  p_q[PIPE_DEPTH_RED128T64-1],r_o,rexp);
 `endif
      end
    end
  end
`endif

endmodule


module dsp_red128t64
  #(
    parameter AB_DUAL_REG      = 0,
    parameter C_DUAL_REG       = 0,
    parameter USE_CARRY        = 0,
    parameter NO_DSP           = 0
   )   
  (
   input logic         rst_ni,
   input logic         clk_i,
   input logic         ce_i,
   input logic [47:0]  AB_i,
   input logic [47:0]  C_i,
   input logic         sub_i,
   input logic         nop_i,
   input logic         CARRYCASCIN_i,
   output logic [47:0] P_o,
   output logic        CARRYCASCOUT_o
   );

  logic signed [47:0]  c0_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      c0_q <= C_i;
    end
  end

  if (NO_DSP) begin

  logic signed [47:0]  ab0_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      ab0_q <= AB_i;
    end
  end

  logic signed [47:0]  ab_q;
  logic signed [47:0]  c_q;
  logic                sub_q, nop_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      ab_q <= AB_DUAL_REG ? ab0_q : AB_i;
      c_q <= C_DUAL_REG ? c0_q : C_i;
      sub_q <= sub_i;
      nop_q <= nop_i;
    end
  end

  logic signed [48:0] p_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      p_q <= nop_q ? c_q 
             : sub_q ? (c_q - ab_q - (USE_CARRY & CARRYCASCIN_i)) 
               : (c_q + ab_q + (USE_CARRY & CARRYCASCIN_i));
    end
  end

  assign P_o = p_q[47:0];
  assign CARRYCASCOUT_o = p_q[48];

  end
  else begin

  logic [3:0] alumode;
  logic [4:0] inmode;
  logic [8:0] opmode;
  logic [47:0] dsp48e2_p;
  logic        dsp48e2_carrycascout;

  always_comb begin
    alumode = sub_i ? 4'b0011 : 4'b0000; // Z - (W + X + Y + CIN) : Z + (W + X + Y + CIN)
    inmode = AB_DUAL_REG ? 5'b00000 : 5'b10001;
    opmode = {
              2'b00,                                // W <- 0
              3'b011,                               // Z <- C
              2'b00,                                // Y <- 0
              nop_i ? 2'b00 : 2'b11                 // X <- 0 : A:B
              };
  end

  DSP48E2 
    #(
      .ACASCREG(),
      .ADREG(),
      .ALUMODEREG(),
      .AMULTSEL(),
      .AREG(AB_DUAL_REG ? 2 : 1),
      .AUTORESET_PATDET(),
      .AUTORESET_PRIORITY(),
      .A_INPUT(),
      .BCASCREG(),
      .BMULTSEL(),
      .BREG(AB_DUAL_REG ? 2 : 1),
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
      .MREG(0),
      .OPMODEREG(),
      .PATTERN(),
      .PREADDINSEL(),
      .PREG(),
      .RND(),
      .SEL_MASK(),
      .SEL_PATTERN(),
      .USE_MULT("NONE"),
      .USE_PATTERN_DETECT(),
      .USE_SIMD(),
      .USE_WIDEXOR(),
      .XORSIMD()
      )
  dsp
    (
     .ACOUT(),
     .BCOUT(),
     .CARRYCASCOUT(dsp48e2_carrycascout),
     .CARRYOUT(),
     .MULTSIGNOUT(),
     .OVERFLOW(),
     .P(dsp48e2_p),
     .PATTERNBDETECT(),
     .PATTERNDETECT(),
     .PCOUT(),
     .UNDERFLOW(),
     .XOROUT(),

     .A(AB_i[47:18]),
     .ACIN(30'h0),
     .ALUMODE(alumode[3:0]),
     .B(AB_i[17:0]),
     .BCIN(18'h0),
     .C(C_DUAL_REG ? c0_q : C_i),
     .CARRYCASCIN(CARRYCASCIN_i),
     .CARRYIN(1'b0),
     .CARRYINSEL(USE_CARRY ? 3'b010: 3'b000), // CARRYSCANIN : CARRYIN
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
     .PCIN(),
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

  assign P_o = dsp48e2_p;
  assign CARRYCASCOUT_o = dsp48e2_carrycascout;

  end

endmodule
