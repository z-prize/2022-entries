// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module modaddsub
  import math_pkg::*;
  #(
    parameter NO_DSP_ADD64    = 0,
    parameter NO_DSP_SUB64    = 0,
    parameter NO_DSP_ADDSUB64 = 0,
    parameter CANONICAL       = 0
    )
  (
   input logic         rst_ni,
   input logic         clk_i,
   input               ce_i,
   input               nop_i,

   input logic [63:0]  x_i,
   input logic [63:0]  y_i,

   output logic [63:0] x_add_y_o,
   output logic [63:0] x_sub_y_o
   );

  logic                nop_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      nop_q <= nop_i;
    end
  end

  //
  // add
  //
  
  logic [47:0] add0_p;
  logic        add0_carry, add0_pdetect;
  dsp_modaddsub
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (0),
       .USE_CARRY        (0),
       .PATTERN          ({32'h00000000, 16'h0000}),
       .MASK             ({32'h00000000, 16'hffff}),
       .NO_DSP           (NO_DSP_ADD64) )
  dsp_add0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({y_i[0+:32],16'h0}),
      .C_i({x_i[0+:32],16'h0}),
      .sub_i(1'b0),
      .zero_ab_i(nop_i),
      .zero_c_i(1'b0),
      .CARRYCASCIN_i(),
      .P_o(add0_p),
      .CARRYCASCOUT_o(add0_carry),
      .PATTERNDETECT_o(add0_pdetect)
      );

  logic [47:0] add1_p;
  logic        add1_pdetect;
  dsp_modaddsub
    #( .AB_DUAL_REG      (1),
       .C_DUAL_REG       (1),
       .USE_CARRY        (1),
       .PATTERN          ({16'h0000, 32'hffffffff}),
       .MASK             ({16'hffff, 32'h00000000}),
       .NO_DSP           (NO_DSP_ADD64) )
  dsp_add1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({16'h0,y_i[32+:32]}),
      .C_i({16'h0,x_i[32+:32]}),
      .sub_i(1'b0),
      .zero_ab_i(nop_q),
      .zero_c_i(1'b0),
      .CARRYCASCIN_i(add0_carry),
      .P_o(add1_p),
      .CARRYCASCOUT_o(),
      .PATTERNDETECT_o(add1_pdetect)
      );

  logic        addmod_nop, addmod_nop_q;
  logic        add0_pdetect_q, addmod_2x, addmod_2x_q;
  always_comb begin
    addmod_nop = add1_p[32]==0;
    addmod_2x = add1_pdetect && !add0_pdetect_q;
  end
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      add0_pdetect_q <= add0_pdetect;
      addmod_nop_q <= addmod_nop;
      addmod_2x_q <= addmod_2x;
    end
  end

  logic [47:0] radd0_p;
  logic        radd0_carry;
  logic        addmod0_pdetect;
  dsp_modaddsub
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (0),
       .PATTERN          ({32'h00000000, 16'h0000}),
       .MASK             ({32'h00000000, 16'hffff}),
       .NO_DSP           (NO_DSP_ADDSUB64) )
  dsp_addmod0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({addmod_2x ? 32'hfffffffe : 32'hffffffff,16'h0}),
      .C_i({add0_p[47:16],16'h0}),
      .sub_i(1'b0),
      .zero_ab_i(addmod_nop),
      .zero_c_i(1'b0),
      .CARRYCASCIN_i(),
      .P_o(radd0_p),
      .CARRYCASCOUT_o(radd0_carry),
      .PATTERNDETECT_o(addmod0_pdetect)
      );

  logic [47:0] radd1_p;
  logic        addmod1_pdetect;
  dsp_modaddsub
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (1),
       .PATTERN          ({16'h0000, 32'hffffffff}),
       .MASK             ({16'hffff, 32'h00000000}),
       .NO_DSP           (NO_DSP_ADDSUB64) )
  dsp_addmod1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i(addmod_2x_q ? 48'h1 : 48'h0),
      .C_i({16'h0,add1_p[31:0]}),
      .sub_i(1'b0),
      .zero_ab_i(addmod_nop_q),
      .zero_c_i(1'b0),
      .CARRYCASCIN_i(radd0_carry),
      .P_o(radd1_p),
      .CARRYCASCOUT_o(),
      .PATTERNDETECT_o(addmod1_pdetect)
      );

  logic [47:0] radd0_q;
  logic [31:0] radd0_dec_q;
  logic        addmod0_pdetect_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      addmod0_pdetect_q <= addmod0_pdetect;
      radd0_q <= radd0_p;
      radd0_dec_q <= radd0_p[47:16] + 32'hffffffff;
    end
  end

  always_comb begin
    if (CANONICAL && addmod1_pdetect && !addmod0_pdetect_q) begin
      x_add_y_o = {32'h00000000,radd0_dec_q};
    end
    else begin
      x_add_y_o = {radd1_p[31:0],radd0_q[47:16]};
    end
  end


  //
  // sub
  //

  logic [47:0] sub0_p;
  logic        sub0_carry, sub0_pdetect;
  dsp_modaddsub
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (0),
       .USE_CARRY        (0),
       .PATTERN          ({32'hffffffff, 16'h0000}),
       .MASK             ({32'h00000000, 16'hffff}),
       .NO_DSP           (NO_DSP_SUB64) )
  dsp_sub0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({y_i[0+:32],16'h0}),
      .C_i({x_i[0+:32],16'h0}),
      .sub_i(!nop_i),
      .zero_ab_i(1'b0),
      .zero_c_i(nop_i),
      .CARRYCASCIN_i(),
      .P_o(sub0_p),
      .CARRYCASCOUT_o(sub0_carry),
      .PATTERNDETECT_o(sub0_pdetect)
      );

  logic [47:0] sub1_p;
  logic        sub1_pdetect;        
  dsp_modaddsub
    #( .AB_DUAL_REG      (1),
       .C_DUAL_REG       (1),
       .USE_CARRY        (1),
       .PATTERN          ({16'h0000, 32'h00000000}),
       .MASK             ({16'hffff, 32'h00000000}),
       .NO_DSP           (NO_DSP_SUB64) )
  dsp_sub1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({16'h0,y_i[32+:32]}),
      .C_i({16'h0,x_i[32+:32]}),
      .sub_i(!nop_q),
      .zero_ab_i(1'b0),
      .zero_c_i(nop_q),
      .CARRYCASCIN_i(sub0_carry),
      .P_o(sub1_p),
      .CARRYCASCOUT_o(),
      .PATTERNDETECT_o(sub1_pdetect)
      );

  logic        submod_nop, submod_nop_q;
  logic        sub0_pdetect_q, submod_2x, submod_2x_q;
  always_comb begin
    submod_nop = sub1_p[32]==0;
    submod_2x = sub1_pdetect && !sub0_pdetect_q;
  end
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      sub0_pdetect_q <= sub0_pdetect;
      submod_nop_q <= submod_nop;
      submod_2x_q <= submod_2x;
    end
  end

  logic [47:0] rsub0_p;
  logic        rsub0_carry;
  logic        submod0_pdetect;
  dsp_modaddsub
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (0),
       .PATTERN          ({32'h00000000, 16'h0000}),
       .MASK             ({32'h00000000, 16'hffff}),
       .NO_DSP           (NO_DSP_ADDSUB64) )
  dsp_submod0
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i({submod_2x ? 32'hfffffffe : 32'hffffffff,16'h0}),
      .C_i({sub0_p[47:16],16'h0}),
      .sub_i(!submod_nop),
      .zero_ab_i(submod_nop),
      .zero_c_i(1'b0),
      .CARRYCASCIN_i(),
      .P_o(rsub0_p),
      .CARRYCASCOUT_o(rsub0_carry),
      .PATTERNDETECT_o(submod0_pdetect)
      );

  logic [47:0] rsub1_p;
  logic        submod1_pdetect;
  dsp_modaddsub
    #( .AB_DUAL_REG      (0),
       .C_DUAL_REG       (1),
       .USE_CARRY        (1),
       .PATTERN          ({16'h0000, 32'hffffffff}),
       .MASK             ({16'hffff, 32'h00000000}),
       .NO_DSP           (NO_DSP_ADDSUB64) )
  dsp_submod1
    ( 
      .rst_ni,
      .clk_i,
      .ce_i,
      .AB_i(submod_2x_q ? 48'h1 : 48'h0),
      .C_i({16'h0,sub1_p[31:0]}),
      .sub_i(!submod_nop_q),
      .zero_ab_i(submod_nop_q),
      .zero_c_i(1'b0),
      .CARRYCASCIN_i(rsub0_carry),
      .P_o(rsub1_p),
      .CARRYCASCOUT_o(),
      .PATTERNDETECT_o(submod1_pdetect)
      );

  logic [47:0] rsub0_q;
  logic [31:0] rsub0_dec_q;
  logic        submod0_pdetect_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      submod0_pdetect_q <= submod0_pdetect;
      rsub0_q <= rsub0_p;
      rsub0_dec_q <= rsub0_p[47:16] + 32'hffffffff;
    end
  end

  always_comb begin
    if (CANONICAL && submod1_pdetect && !submod0_pdetect_q) begin
      x_sub_y_o = {32'h00000000,rsub0_dec_q};
    end
    else begin
      x_sub_y_o = {rsub1_p[31:0],rsub0_q[47:16]};
    end
  end

`ifndef SYNTHESIS
  localparam [63:0] M = 64'hffff_ffff_0000_0001;
  logic [PIPE_DEPTH_MODADDSUB-1:0][63:0] x_q, y_q;
  logic [PIPE_DEPTH_MODADDSUB-1:0] nop_i_q;
  logic [PIPE_DEPTH_MODADDSUB-1:0] v_q;
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      v_q <= 0;
    end
    else if (ce_i) begin
      v_q <= {v_q, 1'b1};
      x_q <= {x_q, x_i};
      y_q <= {y_q, y_i};
      nop_i_q <= {nop_i_q, nop_i};
    end
  end
  logic [64:0] addexp, subexp;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      addexp = nop_i_q[PIPE_DEPTH_MODADDSUB-1] ? x_q[PIPE_DEPTH_MODADDSUB-1] : 
               (x_q[PIPE_DEPTH_MODADDSUB-1] + y_q[PIPE_DEPTH_MODADDSUB-1]);
      if (addexp[64]) begin
        addexp -= 1 << 64;
        addexp += 32'hffffffff;
      end
      if (addexp[64]) begin
        addexp -= 1 << 64;
        addexp += 32'hffffffff;
      end
      if (CANONICAL) begin
        addexp %= M;
      end
      subexp = nop_i_q[PIPE_DEPTH_MODADDSUB-1] ? y_q[PIPE_DEPTH_MODADDSUB-1] : 
               (x_q[PIPE_DEPTH_MODADDSUB-1] - y_q[PIPE_DEPTH_MODADDSUB-1]);
      if (subexp[64]) begin
        subexp += 1 << 64;
        subexp -= 32'hffffffff;
      end
      if (subexp[64]) begin
        subexp += 1 << 64;
        subexp -= 32'hffffffff;
      end
      if (CANONICAL) begin
        subexp %= M;
      end
      if (v_q[PIPE_DEPTH_MODADDSUB-1] && (addexp!=x_add_y_o || subexp!=x_sub_y_o)) begin
        $display("v=%b x=%x y=%x nop=%d addact=%x addexp=%x subact=%x subexp=%x ERROR",
		 v_q[PIPE_DEPTH_MODADDSUB-1],
		 x_q[PIPE_DEPTH_MODADDSUB-1],y_q[PIPE_DEPTH_MODADDSUB-1],
                 nop_i_q[PIPE_DEPTH_MODADDSUB-1],
		 x_add_y_o,addexp,x_sub_y_o,subexp);
        $finish;
      end
      else begin
 `ifdef NEVER
	 $display("v=%b x=%x y=%x nop=%d addact=%x addexp=%x subact=%x subexp=%x",
		  v_q[PIPE_DEPTH_MODADDSUB-1],
		  x_q[PIPE_DEPTH_MODADDSUB-1],y_q[PIPE_DEPTH_MODADDSUB-1],
                  nop_i_q[PIPE_DEPTH_MODADDSUB-1],
		  x_add_y_o,addexp,x_sub_y_o,subexp);
 `endif
      end
    end
  end
`endif

endmodule


module dsp_modaddsub
  #(
    parameter AB_DUAL_REG      = 0,
    parameter C_DUAL_REG       = 0,
    parameter USE_CARRY        = 0,
    parameter [47:0] PATTERN   = 0,
    parameter [47:0] MASK      = 0,
    parameter NO_DSP           = 0
   )   
  (
   input logic         rst_ni,
   input logic         clk_i,
   input logic         ce_i,
   input logic [47:0]  AB_i,
   input logic [47:0]  C_i,
   input logic         sub_i,
   input logic         zero_ab_i,
   input logic         zero_c_i,
   input logic         CARRYCASCIN_i,
   output logic [47:0] P_o,
   output logic        CARRYCASCOUT_o,
   output logic        PATTERNDETECT_o        
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
  logic                sub_q, zero_ab_q, zero_c_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      ab_q <= AB_DUAL_REG ? ab0_q : AB_i;
      c_q <= C_DUAL_REG ? c0_q : C_i;
      sub_q <= sub_i;
      zero_ab_q <= zero_ab_i;
      zero_c_q <= zero_c_i;
    end
  end

  logic signed [48:0] p_q;
  always_ff @(posedge clk_i) begin
    if (ce_i) begin
      p_q <= sub_q ? ((zero_c_q ? 0 : c_q) - (zero_ab_q ? 0 : ab_q) - (USE_CARRY & CARRYCASCIN_i)) 
               : ((zero_c_q ? 0 : c_q) + (zero_ab_q ? 0 : ab_q) + (USE_CARRY & CARRYCASCIN_i));
    end
  end

  assign P_o = p_q[47:0];
  assign CARRYCASCOUT_o = p_q[48];
  assign PATTERNDETECT_o = (p_q[47:0] & ~MASK) == (PATTERN & ~MASK);

  end
  else begin

  logic [3:0] alumode;
  logic [4:0] inmode;
  logic [8:0] opmode;
  logic [47:0] dsp48e2_p;
  logic        dsp48e2_carrycascout;
  logic        dsp48e2_patterndetect;

  always_comb begin
    alumode = sub_i ? 4'b0011 : 4'b0000; // Z - (W + X + Y + CIN) : Z + (W + X + Y + CIN)
    inmode = AB_DUAL_REG ? 5'b00000 : 5'b10001;
    opmode = {
              2'b00,                                // W <- 0
              zero_c_i ? 3'b000: 3'b011,            // Z <- 0 : C
              2'b00,                                // Y <- 0
              zero_ab_i ? 2'b00 : 2'b11             // X <- 0 : A:B
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
      .MASK(MASK),
      .MREG(0),
      .OPMODEREG(),
      .PATTERN(PATTERN),
      .PREADDINSEL(),
      .PREG(),
      .RND(),
      .SEL_MASK(),
      .SEL_PATTERN(),
      .USE_MULT("NONE"),
      .USE_PATTERN_DETECT("PATDET"),
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
     .PATTERNDETECT(dsp48e2_patterndetect),
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
  assign PATTERNDETECT_o = dsp48e2_patterndetect;

  end

endmodule
