// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
//`timescale 1ns/1ps
module csr
#(parameter
    C_S_AXI_ADDR_WIDTH = 9,
    C_S_AXI_DATA_WIDTH = 32
)(
    input  wire                          ACLK,
    input  wire                          ARESET,
    input  wire                          ACLK_EN,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input  wire                          AWVALID,
    output wire                          AWREADY,
    input  wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input  wire                          WVALID,
    output wire                          WREADY,
    output wire [1:0]                    BRESP,
    output wire                          BVALID,
    input  wire                          BREADY,
    input  wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input  wire                          ARVALID,
    output wire                          ARREADY,
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                    RRESP,
    output wire                          RVALID,
    input  wire                          RREADY,
    output wire                          interrupt,
    output wire [0:0]                    chicken_bits,
    output wire [63:0]                   axi00_ptr0,
    output wire [63:0]                   axi01_ptr0,
    output wire [63:0]                   axi02_ptr0,
    output wire [63:0]                   axi03_ptr0,
    output wire [63:0]                   axi04_ptr0,
    output wire [63:0]                   axi05_ptr0,
    output wire [63:0]                   axi06_ptr0,
    output wire [63:0]                   axi07_ptr0,
    output wire [63:0]                   axi08_ptr0,
    output wire [63:0]                   axi09_ptr0,
    output wire [63:0]                   axi10_ptr0,
    output wire [63:0]                   axi11_ptr0,
    output wire [63:0]                   axi12_ptr0,
    output wire [63:0]                   axi13_ptr0,
    output wire [63:0]                   axi14_ptr0,
    output wire [63:0]                   axi15_ptr0,
    output wire [63:0]                   axi16_ptr0,
    output wire [63:0]                   axi17_ptr0,
    output wire [63:0]                   axi18_ptr0,
    output wire [63:0]                   axi19_ptr0,
    output wire [63:0]                   axi20_ptr0,
    output wire [63:0]                   axi21_ptr0,
    output wire [63:0]                   axi22_ptr0,
    output wire [63:0]                   axi23_ptr0,
    output wire [63:0]                   axi24_ptr0,
    output wire [63:0]                   axi25_ptr0,
    output wire [63:0]                   axi26_ptr0,
    output wire [63:0]                   axi27_ptr0,
    output wire [63:0]                   axi28_ptr0,
    output wire [63:0]                   axi29_ptr0,
    output wire [63:0]                   axi30_ptr0,
    output wire [63:0]                   axi31_ptr0,
    output wire                          ap_start,
    input  wire                          ap_done,
    input  wire                          ap_ready,
    input  wire                          ap_idle
);
//------------------------Address Info-------------------
// 0x000 : Control signals
//         bit 0  - ap_start (Read/Write/COH)
//         bit 1  - ap_done (Read/COR)
//         bit 2  - ap_idle (Read)
//         bit 3  - ap_ready (Read/COR)
//         bit 7  - auto_restart (Read/Write)
//         bit 9  - interrupt (Read)
//         others - reserved
// 0x004 : Global Interrupt Enable Register
//         bit 0  - Global Interrupt Enable (Read/Write)
//         others - reserved
// 0x008 : IP Interrupt Enable Register (Read/Write)
//         bit 0 - enable ap_done interrupt (Read/Write)
//         bit 1 - enable ap_ready interrupt (Read/Write)
//         others - reserved
// 0x00c : IP Interrupt Status Register (Read/COR)
//         bit 0 - ap_done (Read/COR)
//         bit 1 - ap_ready (Read/COR)
//         others - reserved
// 0x010 : Data signal of chicken_bits
//         bit 0  - chicken_bits[0] (Read/Write)
//         others - reserved
// 0x014 : reserved
// 0x018 : Data signal of axi00_ptr0
//         bit 31~0 - axi00_ptr0[31:0] (Read/Write)
// 0x01c : Data signal of axi00_ptr0
//         bit 31~0 - axi00_ptr0[63:32] (Read/Write)
// 0x020 : reserved
// 0x024 : Data signal of axi01_ptr0
//         bit 31~0 - axi01_ptr0[31:0] (Read/Write)
// 0x028 : Data signal of axi01_ptr0
//         bit 31~0 - axi01_ptr0[63:32] (Read/Write)
// 0x02c : reserved
// 0x030 : Data signal of axi02_ptr0
//         bit 31~0 - axi02_ptr0[31:0] (Read/Write)
// 0x034 : Data signal of axi02_ptr0
//         bit 31~0 - axi02_ptr0[63:32] (Read/Write)
// 0x038 : reserved
// 0x03c : Data signal of axi03_ptr0
//         bit 31~0 - axi03_ptr0[31:0] (Read/Write)
// 0x040 : Data signal of axi03_ptr0
//         bit 31~0 - axi03_ptr0[63:32] (Read/Write)
// 0x044 : reserved
// 0x048 : Data signal of axi04_ptr0
//         bit 31~0 - axi04_ptr0[31:0] (Read/Write)
// 0x04c : Data signal of axi04_ptr0
//         bit 31~0 - axi04_ptr0[63:32] (Read/Write)
// 0x050 : reserved
// 0x054 : Data signal of axi05_ptr0
//         bit 31~0 - axi05_ptr0[31:0] (Read/Write)
// 0x058 : Data signal of axi05_ptr0
//         bit 31~0 - axi05_ptr0[63:32] (Read/Write)
// 0x05c : reserved
// 0x060 : Data signal of axi06_ptr0
//         bit 31~0 - axi06_ptr0[31:0] (Read/Write)
// 0x064 : Data signal of axi06_ptr0
//         bit 31~0 - axi06_ptr0[63:32] (Read/Write)
// 0x068 : reserved
// 0x06c : Data signal of axi07_ptr0
//         bit 31~0 - axi07_ptr0[31:0] (Read/Write)
// 0x070 : Data signal of axi07_ptr0
//         bit 31~0 - axi07_ptr0[63:32] (Read/Write)
// 0x074 : reserved
// 0x078 : Data signal of axi08_ptr0
//         bit 31~0 - axi08_ptr0[31:0] (Read/Write)
// 0x07c : Data signal of axi08_ptr0
//         bit 31~0 - axi08_ptr0[63:32] (Read/Write)
// 0x080 : reserved
// 0x084 : Data signal of axi09_ptr0
//         bit 31~0 - axi09_ptr0[31:0] (Read/Write)
// 0x088 : Data signal of axi09_ptr0
//         bit 31~0 - axi09_ptr0[63:32] (Read/Write)
// 0x08c : reserved
// 0x090 : Data signal of axi10_ptr0
//         bit 31~0 - axi10_ptr0[31:0] (Read/Write)
// 0x094 : Data signal of axi10_ptr0
//         bit 31~0 - axi10_ptr0[63:32] (Read/Write)
// 0x098 : reserved
// 0x09c : Data signal of axi11_ptr0
//         bit 31~0 - axi11_ptr0[31:0] (Read/Write)
// 0x0a0 : Data signal of axi11_ptr0
//         bit 31~0 - axi11_ptr0[63:32] (Read/Write)
// 0x0a4 : reserved
// 0x0a8 : Data signal of axi12_ptr0
//         bit 31~0 - axi12_ptr0[31:0] (Read/Write)
// 0x0ac : Data signal of axi12_ptr0
//         bit 31~0 - axi12_ptr0[63:32] (Read/Write)
// 0x0b0 : reserved
// 0x0b4 : Data signal of axi13_ptr0
//         bit 31~0 - axi13_ptr0[31:0] (Read/Write)
// 0x0b8 : Data signal of axi13_ptr0
//         bit 31~0 - axi13_ptr0[63:32] (Read/Write)
// 0x0bc : reserved
// 0x0c0 : Data signal of axi14_ptr0
//         bit 31~0 - axi14_ptr0[31:0] (Read/Write)
// 0x0c4 : Data signal of axi14_ptr0
//         bit 31~0 - axi14_ptr0[63:32] (Read/Write)
// 0x0c8 : reserved
// 0x0cc : Data signal of axi15_ptr0
//         bit 31~0 - axi15_ptr0[31:0] (Read/Write)
// 0x0d0 : Data signal of axi15_ptr0
//         bit 31~0 - axi15_ptr0[63:32] (Read/Write)
// 0x0d4 : reserved
// 0x0d8 : Data signal of axi16_ptr0
//         bit 31~0 - axi16_ptr0[31:0] (Read/Write)
// 0x0dc : Data signal of axi16_ptr0
//         bit 31~0 - axi16_ptr0[63:32] (Read/Write)
// 0x0e0 : reserved
// 0x0e4 : Data signal of axi17_ptr0
//         bit 31~0 - axi17_ptr0[31:0] (Read/Write)
// 0x0e8 : Data signal of axi17_ptr0
//         bit 31~0 - axi17_ptr0[63:32] (Read/Write)
// 0x0ec : reserved
// 0x0f0 : Data signal of axi18_ptr0
//         bit 31~0 - axi18_ptr0[31:0] (Read/Write)
// 0x0f4 : Data signal of axi18_ptr0
//         bit 31~0 - axi18_ptr0[63:32] (Read/Write)
// 0x0f8 : reserved
// 0x0fc : Data signal of axi19_ptr0
//         bit 31~0 - axi19_ptr0[31:0] (Read/Write)
// 0x100 : Data signal of axi19_ptr0
//         bit 31~0 - axi19_ptr0[63:32] (Read/Write)
// 0x104 : reserved
// 0x108 : Data signal of axi20_ptr0
//         bit 31~0 - axi20_ptr0[31:0] (Read/Write)
// 0x10c : Data signal of axi20_ptr0
//         bit 31~0 - axi20_ptr0[63:32] (Read/Write)
// 0x110 : reserved
// 0x114 : Data signal of axi21_ptr0
//         bit 31~0 - axi21_ptr0[31:0] (Read/Write)
// 0x118 : Data signal of axi21_ptr0
//         bit 31~0 - axi21_ptr0[63:32] (Read/Write)
// 0x11c : reserved
// 0x120 : Data signal of axi22_ptr0
//         bit 31~0 - axi22_ptr0[31:0] (Read/Write)
// 0x124 : Data signal of axi22_ptr0
//         bit 31~0 - axi22_ptr0[63:32] (Read/Write)
// 0x128 : reserved
// 0x12c : Data signal of axi23_ptr0
//         bit 31~0 - axi23_ptr0[31:0] (Read/Write)
// 0x130 : Data signal of axi23_ptr0
//         bit 31~0 - axi23_ptr0[63:32] (Read/Write)
// 0x134 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

//------------------------Parameter----------------------
localparam
    ADDR_AP_CTRL             = 9'h000,
    ADDR_GIE                 = 9'h004,
    ADDR_IER                 = 9'h008,
    ADDR_ISR                 = 9'h00c,
    ADDR_CHICKEN_BITS_DATA_0 = 9'h010,
    ADDR_CHICKEN_BITS_CTRL   = 9'h014,
    ADDR_AXI00_PTR0_DATA_0   = 9'h018,
    ADDR_AXI00_PTR0_DATA_1   = 9'h01c,
    ADDR_AXI00_PTR0_CTRL     = 9'h020,
    ADDR_AXI01_PTR0_DATA_0   = 9'h024,
    ADDR_AXI01_PTR0_DATA_1   = 9'h028,
    ADDR_AXI01_PTR0_CTRL     = 9'h02c,
    ADDR_AXI02_PTR0_DATA_0   = 9'h030,
    ADDR_AXI02_PTR0_DATA_1   = 9'h034,
    ADDR_AXI02_PTR0_CTRL     = 9'h038,
    ADDR_AXI03_PTR0_DATA_0   = 9'h03c,
    ADDR_AXI03_PTR0_DATA_1   = 9'h040,
    ADDR_AXI03_PTR0_CTRL     = 9'h044,
    ADDR_AXI04_PTR0_DATA_0   = 9'h048,
    ADDR_AXI04_PTR0_DATA_1   = 9'h04c,
    ADDR_AXI04_PTR0_CTRL     = 9'h050,
    ADDR_AXI05_PTR0_DATA_0   = 9'h054,
    ADDR_AXI05_PTR0_DATA_1   = 9'h058,
    ADDR_AXI05_PTR0_CTRL     = 9'h05c,
    ADDR_AXI06_PTR0_DATA_0   = 9'h060,
    ADDR_AXI06_PTR0_DATA_1   = 9'h064,
    ADDR_AXI06_PTR0_CTRL     = 9'h068,
    ADDR_AXI07_PTR0_DATA_0   = 9'h06c,
    ADDR_AXI07_PTR0_DATA_1   = 9'h070,
    ADDR_AXI07_PTR0_CTRL     = 9'h074,
    ADDR_AXI08_PTR0_DATA_0   = 9'h078,
    ADDR_AXI08_PTR0_DATA_1   = 9'h07c,
    ADDR_AXI08_PTR0_CTRL     = 9'h080,
    ADDR_AXI09_PTR0_DATA_0   = 9'h084,
    ADDR_AXI09_PTR0_DATA_1   = 9'h088,
    ADDR_AXI09_PTR0_CTRL     = 9'h08c,
    ADDR_AXI10_PTR0_DATA_0   = 9'h090,
    ADDR_AXI10_PTR0_DATA_1   = 9'h094,
    ADDR_AXI10_PTR0_CTRL     = 9'h098,
    ADDR_AXI11_PTR0_DATA_0   = 9'h09c,
    ADDR_AXI11_PTR0_DATA_1   = 9'h0a0,
    ADDR_AXI11_PTR0_CTRL     = 9'h0a4,
    ADDR_AXI12_PTR0_DATA_0   = 9'h0a8,
    ADDR_AXI12_PTR0_DATA_1   = 9'h0ac,
    ADDR_AXI12_PTR0_CTRL     = 9'h0b0,
    ADDR_AXI13_PTR0_DATA_0   = 9'h0b4,
    ADDR_AXI13_PTR0_DATA_1   = 9'h0b8,
    ADDR_AXI13_PTR0_CTRL     = 9'h0bc,
    ADDR_AXI14_PTR0_DATA_0   = 9'h0c0,
    ADDR_AXI14_PTR0_DATA_1   = 9'h0c4,
    ADDR_AXI14_PTR0_CTRL     = 9'h0c8,
    ADDR_AXI15_PTR0_DATA_0   = 9'h0cc,
    ADDR_AXI15_PTR0_DATA_1   = 9'h0d0,
    ADDR_AXI15_PTR0_CTRL     = 9'h0d4,
    ADDR_AXI16_PTR0_DATA_0   = 9'h0d8,
    ADDR_AXI16_PTR0_DATA_1   = 9'h0dc,
    ADDR_AXI16_PTR0_CTRL     = 9'h0e0,
    ADDR_AXI17_PTR0_DATA_0   = 9'h0e4,
    ADDR_AXI17_PTR0_DATA_1   = 9'h0e8,
    ADDR_AXI17_PTR0_CTRL     = 9'h0ec,
    ADDR_AXI18_PTR0_DATA_0   = 9'h0f0,
    ADDR_AXI18_PTR0_DATA_1   = 9'h0f4,
    ADDR_AXI18_PTR0_CTRL     = 9'h0f8,
    ADDR_AXI19_PTR0_DATA_0   = 9'h0fc,
    ADDR_AXI19_PTR0_DATA_1   = 9'h100,
    ADDR_AXI19_PTR0_CTRL     = 9'h104,
    ADDR_AXI20_PTR0_DATA_0   = 9'h108,
    ADDR_AXI20_PTR0_DATA_1   = 9'h10c,
    ADDR_AXI20_PTR0_CTRL     = 9'h110,
    ADDR_AXI21_PTR0_DATA_0   = 9'h114,
    ADDR_AXI21_PTR0_DATA_1   = 9'h118,
    ADDR_AXI21_PTR0_CTRL     = 9'h11c,
    ADDR_AXI22_PTR0_DATA_0   = 9'h120,
    ADDR_AXI22_PTR0_DATA_1   = 9'h124,
    ADDR_AXI22_PTR0_CTRL     = 9'h128,
    ADDR_AXI23_PTR0_DATA_0   = 9'h12c,
    ADDR_AXI23_PTR0_DATA_1   = 9'h130,
    ADDR_AXI23_PTR0_CTRL     = 9'h134,
    ADDR_AXI24_PTR0_DATA_0   = 9'h138,
    ADDR_AXI24_PTR0_DATA_1   = 9'h13c,
    ADDR_AXI24_PTR0_CTRL     = 9'h140,
    ADDR_AXI25_PTR0_DATA_0   = 9'h144,
    ADDR_AXI25_PTR0_DATA_1   = 9'h148,
    ADDR_AXI25_PTR0_CTRL     = 9'h14c,
    ADDR_AXI26_PTR0_DATA_0   = 9'h150,
    ADDR_AXI26_PTR0_DATA_1   = 9'h154,
    ADDR_AXI26_PTR0_CTRL     = 9'h158,
    ADDR_AXI27_PTR0_DATA_0   = 9'h15c,
    ADDR_AXI27_PTR0_DATA_1   = 9'h160,
    ADDR_AXI27_PTR0_CTRL     = 9'h164,
    ADDR_AXI28_PTR0_DATA_0   = 9'h168,
    ADDR_AXI28_PTR0_DATA_1   = 9'h16c,
    ADDR_AXI28_PTR0_CTRL     = 9'h170,
    ADDR_AXI29_PTR0_DATA_0   = 9'h174,
    ADDR_AXI29_PTR0_DATA_1   = 9'h178,
    ADDR_AXI29_PTR0_CTRL     = 9'h17c,
    ADDR_AXI30_PTR0_DATA_0   = 9'h180,
    ADDR_AXI30_PTR0_DATA_1   = 9'h184,
    ADDR_AXI30_PTR0_CTRL     = 9'h188,
    ADDR_AXI31_PTR0_DATA_0   = 9'h18c,
    ADDR_AXI31_PTR0_DATA_1   = 9'h190,
    ADDR_AXI31_PTR0_CTRL     = 9'h194,
    WRIDLE                   = 2'd0,
    WRDATA                   = 2'd1,
    WRRESP                   = 2'd2,
    WRRESET                  = 2'd3,
    RDIDLE                   = 2'd0,
    RDDATA                   = 2'd1,
    RDRESET                  = 2'd2,
    ADDR_BITS                = 9;

//------------------------Local signal-------------------
    reg  [1:0]                    wstate = WRRESET;
    reg  [1:0]                    wnext;
    reg  [ADDR_BITS-1:0]          waddr;
    wire [C_S_AXI_DATA_WIDTH-1:0] wmask;
    wire                          aw_hs;
    wire                          w_hs;
    reg  [1:0]                    rstate = RDRESET;
    reg  [1:0]                    rnext;
    reg  [C_S_AXI_DATA_WIDTH-1:0] rdata;
    wire                          ar_hs;
    wire [ADDR_BITS-1:0]          raddr;
    // internal registers
    reg                           int_ap_idle;
    reg                           int_ap_ready = 1'b0;
    wire                          task_ap_ready;
    reg                           int_ap_done = 1'b0;
    wire                          task_ap_done;
    reg                           int_task_ap_done = 1'b0;
    reg                           int_ap_start = 1'b0;
    reg                           int_interrupt = 1'b0;
    reg                           int_auto_restart = 1'b0;
    reg                           auto_restart_status = 1'b0;
    wire                          auto_restart_done;
    reg                           int_gie = 1'b0;
    reg  [1:0]                    int_ier = 2'b0;
    reg  [1:0]                    int_isr = 2'b0;
    reg  [0:0]                    int_chicken_bits = 'b0;
    reg  [63:0]                   int_axi00_ptr0 = 'b0;
    reg  [63:0]                   int_axi01_ptr0 = 'b0;
    reg  [63:0]                   int_axi02_ptr0 = 'b0;
    reg  [63:0]                   int_axi03_ptr0 = 'b0;
    reg  [63:0]                   int_axi04_ptr0 = 'b0;
    reg  [63:0]                   int_axi05_ptr0 = 'b0;
    reg  [63:0]                   int_axi06_ptr0 = 'b0;
    reg  [63:0]                   int_axi07_ptr0 = 'b0;
    reg  [63:0]                   int_axi08_ptr0 = 'b0;
    reg  [63:0]                   int_axi09_ptr0 = 'b0;
    reg  [63:0]                   int_axi10_ptr0 = 'b0;
    reg  [63:0]                   int_axi11_ptr0 = 'b0;
    reg  [63:0]                   int_axi12_ptr0 = 'b0;
    reg  [63:0]                   int_axi13_ptr0 = 'b0;
    reg  [63:0]                   int_axi14_ptr0 = 'b0;
    reg  [63:0]                   int_axi15_ptr0 = 'b0;
    reg  [63:0]                   int_axi16_ptr0 = 'b0;
    reg  [63:0]                   int_axi17_ptr0 = 'b0;
    reg  [63:0]                   int_axi18_ptr0 = 'b0;
    reg  [63:0]                   int_axi19_ptr0 = 'b0;
    reg  [63:0]                   int_axi20_ptr0 = 'b0;
    reg  [63:0]                   int_axi21_ptr0 = 'b0;
    reg  [63:0]                   int_axi22_ptr0 = 'b0;
    reg  [63:0]                   int_axi23_ptr0 = 'b0;
    reg  [63:0]                   int_axi24_ptr0 = 'b0;
    reg  [63:0]                   int_axi25_ptr0 = 'b0;
    reg  [63:0]                   int_axi26_ptr0 = 'b0;
    reg  [63:0]                   int_axi27_ptr0 = 'b0;
    reg  [63:0]                   int_axi28_ptr0 = 'b0;
    reg  [63:0]                   int_axi29_ptr0 = 'b0;
    reg  [63:0]                   int_axi30_ptr0 = 'b0;
    reg  [63:0]                   int_axi31_ptr0 = 'b0;

//------------------------Instantiation------------------


//------------------------AXI write fsm------------------
assign AWREADY = (wstate == WRIDLE);
assign WREADY  = (wstate == WRDATA);
assign BRESP   = 2'b00;  // OKAY
assign BVALID  = (wstate == WRRESP);
assign wmask   = { {8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
assign aw_hs   = AWVALID & AWREADY;
assign w_hs    = WVALID & WREADY;

// wstate
always @(posedge ACLK) begin
    if (ARESET)
        wstate <= WRRESET;
    else if (ACLK_EN)
        wstate <= wnext;
end

// wnext
always @(*) begin
    case (wstate)
        WRIDLE:
            if (AWVALID)
                wnext = WRDATA;
            else
                wnext = WRIDLE;
        WRDATA:
            if (WVALID)
                wnext = WRRESP;
            else
                wnext = WRDATA;
        WRRESP:
            if (BREADY)
                wnext = WRIDLE;
            else
                wnext = WRRESP;
        default:
            wnext = WRIDLE;
    endcase
end

// waddr
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (aw_hs)
            waddr <= AWADDR[ADDR_BITS-1:0];
    end
end

//------------------------AXI read fsm-------------------
assign ARREADY = (rstate == RDIDLE);
assign RDATA   = rdata;
assign RRESP   = 2'b00;  // OKAY
assign RVALID  = (rstate == RDDATA);
assign ar_hs   = ARVALID & ARREADY;
assign raddr   = ARADDR[ADDR_BITS-1:0];

// rstate
always @(posedge ACLK) begin
    if (ARESET)
        rstate <= RDRESET;
    else if (ACLK_EN)
        rstate <= rnext;
end

// rnext
always @(*) begin
    case (rstate)
        RDIDLE:
            if (ARVALID)
                rnext = RDDATA;
            else
                rnext = RDIDLE;
        RDDATA:
            if (RREADY & RVALID)
                rnext = RDIDLE;
            else
                rnext = RDDATA;
        default:
            rnext = RDIDLE;
    endcase
end

// rdata
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (ar_hs) begin
            rdata <= 'b0;
            case (raddr)
                ADDR_AP_CTRL: begin
                    rdata[0] <= int_ap_start;
                    rdata[1] <= int_task_ap_done;
                    rdata[2] <= int_ap_idle;
                    rdata[3] <= int_ap_ready;
                    rdata[7] <= int_auto_restart;
                    rdata[9] <= int_interrupt;
                end
                ADDR_GIE: begin
                    rdata <= int_gie;
                end
                ADDR_IER: begin
                    rdata <= int_ier;
                end
                ADDR_ISR: begin
                    rdata <= int_isr;
                end
                ADDR_CHICKEN_BITS_DATA_0: begin
                    rdata <= int_chicken_bits[0:0];
                end
                ADDR_AXI00_PTR0_DATA_0: begin
                    rdata <= int_axi00_ptr0[31:0];
                end
                ADDR_AXI00_PTR0_DATA_1: begin
                    rdata <= int_axi00_ptr0[63:32];
                end
                ADDR_AXI01_PTR0_DATA_0: begin
                    rdata <= int_axi01_ptr0[31:0];
                end
                ADDR_AXI01_PTR0_DATA_1: begin
                    rdata <= int_axi01_ptr0[63:32];
                end
                ADDR_AXI02_PTR0_DATA_0: begin
                    rdata <= int_axi02_ptr0[31:0];
                end
                ADDR_AXI02_PTR0_DATA_1: begin
                    rdata <= int_axi02_ptr0[63:32];
                end
                ADDR_AXI03_PTR0_DATA_0: begin
                    rdata <= int_axi03_ptr0[31:0];
                end
                ADDR_AXI03_PTR0_DATA_1: begin
                    rdata <= int_axi03_ptr0[63:32];
                end
                ADDR_AXI04_PTR0_DATA_0: begin
                    rdata <= int_axi04_ptr0[31:0];
                end
                ADDR_AXI04_PTR0_DATA_1: begin
                    rdata <= int_axi04_ptr0[63:32];
                end
                ADDR_AXI05_PTR0_DATA_0: begin
                    rdata <= int_axi05_ptr0[31:0];
                end
                ADDR_AXI05_PTR0_DATA_1: begin
                    rdata <= int_axi05_ptr0[63:32];
                end
                ADDR_AXI06_PTR0_DATA_0: begin
                    rdata <= int_axi06_ptr0[31:0];
                end
                ADDR_AXI06_PTR0_DATA_1: begin
                    rdata <= int_axi06_ptr0[63:32];
                end
                ADDR_AXI07_PTR0_DATA_0: begin
                    rdata <= int_axi07_ptr0[31:0];
                end
                ADDR_AXI07_PTR0_DATA_1: begin
                    rdata <= int_axi07_ptr0[63:32];
                end
                ADDR_AXI08_PTR0_DATA_0: begin
                    rdata <= int_axi08_ptr0[31:0];
                end
                ADDR_AXI08_PTR0_DATA_1: begin
                    rdata <= int_axi08_ptr0[63:32];
                end
                ADDR_AXI09_PTR0_DATA_0: begin
                    rdata <= int_axi09_ptr0[31:0];
                end
                ADDR_AXI09_PTR0_DATA_1: begin
                    rdata <= int_axi09_ptr0[63:32];
                end
                ADDR_AXI10_PTR0_DATA_0: begin
                    rdata <= int_axi10_ptr0[31:0];
                end
                ADDR_AXI10_PTR0_DATA_1: begin
                    rdata <= int_axi10_ptr0[63:32];
                end
                ADDR_AXI11_PTR0_DATA_0: begin
                    rdata <= int_axi11_ptr0[31:0];
                end
                ADDR_AXI11_PTR0_DATA_1: begin
                    rdata <= int_axi11_ptr0[63:32];
                end
                ADDR_AXI12_PTR0_DATA_0: begin
                    rdata <= int_axi12_ptr0[31:0];
                end
                ADDR_AXI12_PTR0_DATA_1: begin
                    rdata <= int_axi12_ptr0[63:32];
                end
                ADDR_AXI13_PTR0_DATA_0: begin
                    rdata <= int_axi13_ptr0[31:0];
                end
                ADDR_AXI13_PTR0_DATA_1: begin
                    rdata <= int_axi13_ptr0[63:32];
                end
                ADDR_AXI14_PTR0_DATA_0: begin
                    rdata <= int_axi14_ptr0[31:0];
                end
                ADDR_AXI14_PTR0_DATA_1: begin
                    rdata <= int_axi14_ptr0[63:32];
                end
                ADDR_AXI15_PTR0_DATA_0: begin
                    rdata <= int_axi15_ptr0[31:0];
                end
                ADDR_AXI15_PTR0_DATA_1: begin
                    rdata <= int_axi15_ptr0[63:32];
                end
                ADDR_AXI16_PTR0_DATA_0: begin
                    rdata <= int_axi16_ptr0[31:0];
                end
                ADDR_AXI16_PTR0_DATA_1: begin
                    rdata <= int_axi16_ptr0[63:32];
                end
                ADDR_AXI17_PTR0_DATA_0: begin
                    rdata <= int_axi17_ptr0[31:0];
                end
                ADDR_AXI17_PTR0_DATA_1: begin
                    rdata <= int_axi17_ptr0[63:32];
                end
                ADDR_AXI18_PTR0_DATA_0: begin
                    rdata <= int_axi18_ptr0[31:0];
                end
                ADDR_AXI18_PTR0_DATA_1: begin
                    rdata <= int_axi18_ptr0[63:32];
                end
                ADDR_AXI19_PTR0_DATA_0: begin
                    rdata <= int_axi19_ptr0[31:0];
                end
                ADDR_AXI19_PTR0_DATA_1: begin
                    rdata <= int_axi19_ptr0[63:32];
                end
                ADDR_AXI20_PTR0_DATA_0: begin
                    rdata <= int_axi20_ptr0[31:0];
                end
                ADDR_AXI20_PTR0_DATA_1: begin
                    rdata <= int_axi20_ptr0[63:32];
                end
                ADDR_AXI21_PTR0_DATA_0: begin
                    rdata <= int_axi21_ptr0[31:0];
                end
                ADDR_AXI21_PTR0_DATA_1: begin
                    rdata <= int_axi21_ptr0[63:32];
                end
                ADDR_AXI22_PTR0_DATA_0: begin
                    rdata <= int_axi22_ptr0[31:0];
                end
                ADDR_AXI22_PTR0_DATA_1: begin
                    rdata <= int_axi22_ptr0[63:32];
                end
                ADDR_AXI23_PTR0_DATA_0: begin
                    rdata <= int_axi23_ptr0[31:0];
                end
                ADDR_AXI23_PTR0_DATA_1: begin
                    rdata <= int_axi23_ptr0[63:32];
                end
                ADDR_AXI24_PTR0_DATA_0: begin
                    rdata <= int_axi24_ptr0[31:0];
                end
                ADDR_AXI24_PTR0_DATA_1: begin
                    rdata <= int_axi24_ptr0[63:32];
                end
                ADDR_AXI25_PTR0_DATA_0: begin
                    rdata <= int_axi25_ptr0[31:0];
                end
                ADDR_AXI25_PTR0_DATA_1: begin
                    rdata <= int_axi25_ptr0[63:32];
                end
                ADDR_AXI26_PTR0_DATA_0: begin
                    rdata <= int_axi26_ptr0[31:0];
                end
                ADDR_AXI26_PTR0_DATA_1: begin
                    rdata <= int_axi26_ptr0[63:32];
                end
                ADDR_AXI27_PTR0_DATA_0: begin
                    rdata <= int_axi27_ptr0[31:0];
                end
                ADDR_AXI27_PTR0_DATA_1: begin
                    rdata <= int_axi27_ptr0[63:32];
                end
                ADDR_AXI28_PTR0_DATA_0: begin
                    rdata <= int_axi28_ptr0[31:0];
                end
                ADDR_AXI28_PTR0_DATA_1: begin
                    rdata <= int_axi28_ptr0[63:32];
                end
                ADDR_AXI29_PTR0_DATA_0: begin
                    rdata <= int_axi29_ptr0[31:0];
                end
                ADDR_AXI29_PTR0_DATA_1: begin
                    rdata <= int_axi29_ptr0[63:32];
                end
                ADDR_AXI30_PTR0_DATA_0: begin
                    rdata <= int_axi30_ptr0[31:0];
                end
                ADDR_AXI30_PTR0_DATA_1: begin
                    rdata <= int_axi30_ptr0[63:32];
                end
                ADDR_AXI31_PTR0_DATA_0: begin
                    rdata <= int_axi31_ptr0[31:0];
                end
                ADDR_AXI31_PTR0_DATA_1: begin
                    rdata <= int_axi31_ptr0[63:32];
                end
            endcase
        end
    end
end


//------------------------Register logic-----------------
assign interrupt         = int_interrupt;
assign ap_start          = int_ap_start;
assign task_ap_done      = (ap_done && !auto_restart_status) || auto_restart_done;
assign task_ap_ready     = ap_ready && !int_auto_restart;
assign auto_restart_done = auto_restart_status && (ap_idle && !int_ap_idle);
assign chicken_bits      = int_chicken_bits;
assign axi00_ptr0        = int_axi00_ptr0;
assign axi01_ptr0        = int_axi01_ptr0;
assign axi02_ptr0        = int_axi02_ptr0;
assign axi03_ptr0        = int_axi03_ptr0;
assign axi04_ptr0        = int_axi04_ptr0;
assign axi05_ptr0        = int_axi05_ptr0;
assign axi06_ptr0        = int_axi06_ptr0;
assign axi07_ptr0        = int_axi07_ptr0;
assign axi08_ptr0        = int_axi08_ptr0;
assign axi09_ptr0        = int_axi09_ptr0;
assign axi10_ptr0        = int_axi10_ptr0;
assign axi11_ptr0        = int_axi11_ptr0;
assign axi12_ptr0        = int_axi12_ptr0;
assign axi13_ptr0        = int_axi13_ptr0;
assign axi14_ptr0        = int_axi14_ptr0;
assign axi15_ptr0        = int_axi15_ptr0;
assign axi16_ptr0        = int_axi16_ptr0;
assign axi17_ptr0        = int_axi17_ptr0;
assign axi18_ptr0        = int_axi18_ptr0;
assign axi19_ptr0        = int_axi19_ptr0;
assign axi20_ptr0        = int_axi20_ptr0;
assign axi21_ptr0        = int_axi21_ptr0;
assign axi22_ptr0        = int_axi22_ptr0;
assign axi23_ptr0        = int_axi23_ptr0;
assign axi24_ptr0        = int_axi24_ptr0;
assign axi25_ptr0        = int_axi25_ptr0;
assign axi26_ptr0        = int_axi26_ptr0;
assign axi27_ptr0        = int_axi27_ptr0;
assign axi28_ptr0        = int_axi28_ptr0;
assign axi29_ptr0        = int_axi29_ptr0;
assign axi30_ptr0        = int_axi30_ptr0;
assign axi31_ptr0        = int_axi31_ptr0;
// int_interrupt
always @(posedge ACLK) begin
    if (ARESET)
        int_interrupt <= 1'b0;
    else if (ACLK_EN) begin
        if (int_gie && (|int_isr))
            int_interrupt <= 1'b1;
        else
            int_interrupt <= 1'b0;
    end
end

// int_ap_start
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
        else if (ap_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_done <= ap_done;
    end
end

// int_task_ap_done
always @(posedge ACLK) begin
    if (ARESET)
        int_task_ap_done <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_done)
            int_task_ap_done <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_task_ap_done <= 1'b0; // clear on read
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_idle <= ap_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
        if (task_ap_ready)
            int_ap_ready <= 1'b1;
        else if (ar_hs && raddr == ADDR_AP_CTRL)
            int_ap_ready <= 1'b0;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <=  WDATA[7];
    end
end

// auto_restart_status
always @(posedge ACLK) begin
    if (ARESET)
        auto_restart_status <= 1'b0;
    else if (ACLK_EN) begin
        if (int_auto_restart)
            auto_restart_status <= 1'b1;
        else if (ap_idle)
            auto_restart_status <= 1'b0;
    end
end

// int_gie
always @(posedge ACLK) begin
    if (ARESET)
        int_gie <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_GIE && WSTRB[0])
            int_gie <= WDATA[0];
    end
end

// int_ier
always @(posedge ACLK) begin
    if (ARESET)
        int_ier <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_IER && WSTRB[0])
            int_ier <= WDATA[1:0];
    end
end

// int_isr[0]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[0] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[0] & ap_done)
            int_isr[0] <= 1'b1;
        else if (ar_hs && raddr == ADDR_ISR)
            int_isr[0] <= 1'b0; // clear on read
    end
end

// int_isr[1]
always @(posedge ACLK) begin
    if (ARESET)
        int_isr[1] <= 1'b0;
    else if (ACLK_EN) begin
        if (int_ier[1] & ap_ready)
            int_isr[1] <= 1'b1;
        else if (ar_hs && raddr == ADDR_ISR)
            int_isr[1] <= 1'b0; // clear on read
    end
end

// int_chicken_bits[0:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_chicken_bits[0:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_CHICKEN_BITS_DATA_0)
            int_chicken_bits[0:0] <= (WDATA[31:0] & wmask) | (int_chicken_bits[0:0] & ~wmask);
    end
end

// int_axi00_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi00_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI00_PTR0_DATA_0)
            int_axi00_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi00_ptr0[31:0] & ~wmask);
    end
end

// int_axi00_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi00_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI00_PTR0_DATA_1)
            int_axi00_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi00_ptr0[63:32] & ~wmask);
    end
end

// int_axi01_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi01_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI01_PTR0_DATA_0)
            int_axi01_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi01_ptr0[31:0] & ~wmask);
    end
end

// int_axi01_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi01_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI01_PTR0_DATA_1)
            int_axi01_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi01_ptr0[63:32] & ~wmask);
    end
end

// int_axi02_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi02_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI02_PTR0_DATA_0)
            int_axi02_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi02_ptr0[31:0] & ~wmask);
    end
end

// int_axi02_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi02_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI02_PTR0_DATA_1)
            int_axi02_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi02_ptr0[63:32] & ~wmask);
    end
end

// int_axi03_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi03_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI03_PTR0_DATA_0)
            int_axi03_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi03_ptr0[31:0] & ~wmask);
    end
end

// int_axi03_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi03_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI03_PTR0_DATA_1)
            int_axi03_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi03_ptr0[63:32] & ~wmask);
    end
end

// int_axi04_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi04_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI04_PTR0_DATA_0)
            int_axi04_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi04_ptr0[31:0] & ~wmask);
    end
end

// int_axi04_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi04_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI04_PTR0_DATA_1)
            int_axi04_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi04_ptr0[63:32] & ~wmask);
    end
end

// int_axi05_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi05_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI05_PTR0_DATA_0)
            int_axi05_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi05_ptr0[31:0] & ~wmask);
    end
end

// int_axi05_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi05_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI05_PTR0_DATA_1)
            int_axi05_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi05_ptr0[63:32] & ~wmask);
    end
end

// int_axi06_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi06_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI06_PTR0_DATA_0)
            int_axi06_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi06_ptr0[31:0] & ~wmask);
    end
end

// int_axi06_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi06_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI06_PTR0_DATA_1)
            int_axi06_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi06_ptr0[63:32] & ~wmask);
    end
end

// int_axi07_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi07_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI07_PTR0_DATA_0)
            int_axi07_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi07_ptr0[31:0] & ~wmask);
    end
end

// int_axi07_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi07_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI07_PTR0_DATA_1)
            int_axi07_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi07_ptr0[63:32] & ~wmask);
    end
end

// int_axi08_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi08_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI08_PTR0_DATA_0)
            int_axi08_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi08_ptr0[31:0] & ~wmask);
    end
end

// int_axi08_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi08_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI08_PTR0_DATA_1)
            int_axi08_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi08_ptr0[63:32] & ~wmask);
    end
end

// int_axi09_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi09_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI09_PTR0_DATA_0)
            int_axi09_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi09_ptr0[31:0] & ~wmask);
    end
end

// int_axi09_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi09_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI09_PTR0_DATA_1)
            int_axi09_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi09_ptr0[63:32] & ~wmask);
    end
end

// int_axi10_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi10_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI10_PTR0_DATA_0)
            int_axi10_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi10_ptr0[31:0] & ~wmask);
    end
end

// int_axi10_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi10_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI10_PTR0_DATA_1)
            int_axi10_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi10_ptr0[63:32] & ~wmask);
    end
end

// int_axi11_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi11_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI11_PTR0_DATA_0)
            int_axi11_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi11_ptr0[31:0] & ~wmask);
    end
end

// int_axi11_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi11_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI11_PTR0_DATA_1)
            int_axi11_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi11_ptr0[63:32] & ~wmask);
    end
end

// int_axi12_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi12_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI12_PTR0_DATA_0)
            int_axi12_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi12_ptr0[31:0] & ~wmask);
    end
end

// int_axi12_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi12_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI12_PTR0_DATA_1)
            int_axi12_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi12_ptr0[63:32] & ~wmask);
    end
end

// int_axi13_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi13_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI13_PTR0_DATA_0)
            int_axi13_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi13_ptr0[31:0] & ~wmask);
    end
end

// int_axi13_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi13_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI13_PTR0_DATA_1)
            int_axi13_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi13_ptr0[63:32] & ~wmask);
    end
end

// int_axi14_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi14_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI14_PTR0_DATA_0)
            int_axi14_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi14_ptr0[31:0] & ~wmask);
    end
end

// int_axi14_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi14_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI14_PTR0_DATA_1)
            int_axi14_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi14_ptr0[63:32] & ~wmask);
    end
end

// int_axi15_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi15_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI15_PTR0_DATA_0)
            int_axi15_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi15_ptr0[31:0] & ~wmask);
    end
end

// int_axi15_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi15_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI15_PTR0_DATA_1)
            int_axi15_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi15_ptr0[63:32] & ~wmask);
    end
end

// int_axi16_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi16_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI16_PTR0_DATA_0)
            int_axi16_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi16_ptr0[31:0] & ~wmask);
    end
end

// int_axi16_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi16_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI16_PTR0_DATA_1)
            int_axi16_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi16_ptr0[63:32] & ~wmask);
    end
end

// int_axi17_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi17_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI17_PTR0_DATA_0)
            int_axi17_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi17_ptr0[31:0] & ~wmask);
    end
end

// int_axi17_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi17_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI17_PTR0_DATA_1)
            int_axi17_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi17_ptr0[63:32] & ~wmask);
    end
end

// int_axi18_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi18_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI18_PTR0_DATA_0)
            int_axi18_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi18_ptr0[31:0] & ~wmask);
    end
end

// int_axi18_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi18_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI18_PTR0_DATA_1)
            int_axi18_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi18_ptr0[63:32] & ~wmask);
    end
end

// int_axi19_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi19_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI19_PTR0_DATA_0)
            int_axi19_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi19_ptr0[31:0] & ~wmask);
    end
end

// int_axi19_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi19_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI19_PTR0_DATA_1)
            int_axi19_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi19_ptr0[63:32] & ~wmask);
    end
end

// int_axi20_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi20_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI20_PTR0_DATA_0)
            int_axi20_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi20_ptr0[31:0] & ~wmask);
    end
end

// int_axi20_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi20_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI20_PTR0_DATA_1)
            int_axi20_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi20_ptr0[63:32] & ~wmask);
    end
end

// int_axi21_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi21_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI21_PTR0_DATA_0)
            int_axi21_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi21_ptr0[31:0] & ~wmask);
    end
end

// int_axi21_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi21_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI21_PTR0_DATA_1)
            int_axi21_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi21_ptr0[63:32] & ~wmask);
    end
end

// int_axi22_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi22_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI22_PTR0_DATA_0)
            int_axi22_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi22_ptr0[31:0] & ~wmask);
    end
end

// int_axi22_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi22_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI22_PTR0_DATA_1)
            int_axi22_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi22_ptr0[63:32] & ~wmask);
    end
end

// int_axi23_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi23_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI23_PTR0_DATA_0)
            int_axi23_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi23_ptr0[31:0] & ~wmask);
    end
end

// int_axi23_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi23_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI23_PTR0_DATA_1)
            int_axi23_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi23_ptr0[63:32] & ~wmask);
    end
end

// int_axi24_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi24_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI24_PTR0_DATA_0)
            int_axi24_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi24_ptr0[31:0] & ~wmask);
    end
end

// int_axi24_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi24_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI24_PTR0_DATA_1)
            int_axi24_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi24_ptr0[63:32] & ~wmask);
    end
end

// int_axi25_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi25_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI25_PTR0_DATA_0)
            int_axi25_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi25_ptr0[31:0] & ~wmask);
    end
end

// int_axi25_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi25_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI25_PTR0_DATA_1)
            int_axi25_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi25_ptr0[63:32] & ~wmask);
    end
end

// int_axi26_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi26_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI26_PTR0_DATA_0)
            int_axi26_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi26_ptr0[31:0] & ~wmask);
    end
end

// int_axi26_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi26_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI26_PTR0_DATA_1)
            int_axi26_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi26_ptr0[63:32] & ~wmask);
    end
end

// int_axi27_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi27_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI27_PTR0_DATA_0)
            int_axi27_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi27_ptr0[31:0] & ~wmask);
    end
end

// int_axi27_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi27_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI27_PTR0_DATA_1)
            int_axi27_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi27_ptr0[63:32] & ~wmask);
    end
end

// int_axi28_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi28_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI28_PTR0_DATA_0)
            int_axi28_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi28_ptr0[31:0] & ~wmask);
    end
end

// int_axi28_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi28_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI28_PTR0_DATA_1)
            int_axi28_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi28_ptr0[63:32] & ~wmask);
    end
end

// int_axi29_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi29_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI29_PTR0_DATA_0)
            int_axi29_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi29_ptr0[31:0] & ~wmask);
    end
end

// int_axi29_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi29_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI29_PTR0_DATA_1)
            int_axi29_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi29_ptr0[63:32] & ~wmask);
    end
end

// int_axi30_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi30_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI30_PTR0_DATA_0)
            int_axi30_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi30_ptr0[31:0] & ~wmask);
    end
end

// int_axi30_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi30_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI30_PTR0_DATA_1)
            int_axi30_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi30_ptr0[63:32] & ~wmask);
    end
end

// int_axi31_ptr0[31:0]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi31_ptr0[31:0] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI31_PTR0_DATA_0)
            int_axi31_ptr0[31:0] <= (WDATA[31:0] & wmask) | (int_axi31_ptr0[31:0] & ~wmask);
    end
end

// int_axi31_ptr0[63:32]
always @(posedge ACLK) begin
    if (ARESET)
        int_axi31_ptr0[63:32] <= 0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AXI31_PTR0_DATA_1)
            int_axi31_ptr0[63:32] <= (WDATA[31:0] & wmask) | (int_axi31_ptr0[63:32] & ~wmask);
    end
end

//synthesis translate_off
always @(posedge ACLK) begin
    if (ACLK_EN) begin
        if (int_gie & ~int_isr[0] & int_ier[0] & ap_done)
            $display ("// Interrupt Monitor : interrupt for ap_done detected @ \"%0t\"", $time);
        if (int_gie & ~int_isr[1] & int_ier[1] & ap_ready)
            $display ("// Interrupt Monitor : interrupt for ap_ready detected @ \"%0t\"", $time);
    end
end
//synthesis translate_on

//------------------------Memory logic-------------------

endmodule
