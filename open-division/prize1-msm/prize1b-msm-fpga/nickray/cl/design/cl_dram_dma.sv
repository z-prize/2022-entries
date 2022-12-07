// Amazon FPGA Hardware Development Kit
//
// Copyright 2016 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Amazon Software License (the "License"). You may not use
// this file except in compliance with the License. A copy of the License is
// located at
//
//    http://aws.amazon.com/asl/
//
// or in the "license" file accompanying this file. This file is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
// implied. See the License for the specific language governing permissions and
// limitations under the License.

// DDR address space
// The addressing uses ROW/COLUMN/BANK mapping of AXI address to DRAM Row/Col/BankGroup

module cl_dram_dma #(parameter NUM_DDR=4) 

(
   `include "cl_ports.vh"

);

`include "cl_common_defines.vh"      // CL Defines for all examples
`include "cl_id_defines.vh"          // Defines for ID0 and ID1 (PCI ID's)
`include "cl_dram_dma_defines.vh"

// TIE OFF ALL UNUSED INTERFACES
// Including all the unused interface to tie off
// This list is put in the top of the fie to remind
// developers to remve the specific interfaces
// that the CL will use

`include "unused_flr_template.inc"
// `include "unused_ddr_a_b_d_template.inc"
// `include "unused_ddr_c_template.inc"
// `include "unused_pcim_template.inc"
// `include "unused_dma_pcis_template.inc"
`include "unused_cl_sda_template.inc"
`include "unused_sh_bar1_template.inc"
`include "unused_apppf_irq_template.inc"

// Define the addition pipeline stag
// needed to close timing for the various
// place where ATG (Automatic Test Generator)
// is defined
   
   localparam NUM_CFG_STGS_CL_DDR_ATG = 8;
   localparam NUM_CFG_STGS_SH_DDR_ATG = 4;
   localparam NUM_CFG_STGS_PCIE_ATG = 4;

// To reduce RTL simulation time, only 8KiB of
// each external DRAM is scrubbed in simulations

`ifdef SIM
   localparam DDR_SCRB_MAX_ADDR = 64'h1FFF;
`else   
   localparam DDR_SCRB_MAX_ADDR = 64'h3FFFFFFFF; //16GB 
`endif
   localparam DDR_SCRB_BURST_LEN_MINUS1 = 15;

`ifdef NO_CL_TST_SCRUBBER
   localparam NO_SCRB_INST = 1;
`else
   localparam NO_SCRB_INST = 0;
`endif   

logic clk;
(* dont_touch = "true" *) logic pipe_rst_n;
logic pre_sync_rst_n;
(* dont_touch = "true" *) logic sync_rst_n;
logic sh_cl_flr_assert_q;

logic [3:0] all_ddr_scrb_done;
logic [3:0] all_ddr_is_ready;
logic [2:0] lcl_sh_cl_ddr_is_ready;

logic dbg_scrb_en;
logic [2:0] dbg_scrb_mem_sel;

//---------------------------- 
// End Internal signals
//----------------------------

assign cl_sh_status0 = 32'h0;
assign cl_sh_status1 = 32'h0;

assign cl_sh_id0[31:0] = `CL_SH_ID0;
assign cl_sh_id1[31:0] = `CL_SH_ID1;

// Unused 'full' signals
assign cl_sh_dma_rd_full  = 1'b0;
assign cl_sh_dma_wr_full  = 1'b0;

assign clk = clk_main_a0;

//reset synchronizer
lib_pipe #(.WIDTH(1), .STAGES(4)) PIPE_RST_N (.clk(clk), .rst_n(1'b1), .in_bus(rst_main_n), .out_bus(pipe_rst_n));
   
always_ff @(negedge pipe_rst_n or posedge clk)
   if (!pipe_rst_n)
   begin
      pre_sync_rst_n <= 0;
      sync_rst_n <= 0;
   end
   else
   begin
      pre_sync_rst_n <= 1;
      sync_rst_n <= pre_sync_rst_n;
   end









logic [1-1:0] rst;

always_ff@(posedge clk_main_a0)
    rst <= ~sync_rst_n;












//                AAA               VVVVVVVV           VVVVVVVVMMMMMMMM               MMMMMMMMMMMMMMMM               MMMMMMMM
//               A:::A              V::::::V           V::::::VM:::::::M             M:::::::MM:::::::M             M:::::::M
//              A:::::A             V::::::V           V::::::VM::::::::M           M::::::::MM::::::::M           M::::::::M
//             A:::::::A            V::::::V           V::::::VM:::::::::M         M:::::::::MM:::::::::M         M:::::::::M
//            A:::::::::A            V:::::V           V:::::V M::::::::::M       M::::::::::MM::::::::::M       M::::::::::M
//           A:::::A:::::A            V:::::V         V:::::V  M:::::::::::M     M:::::::::::MM:::::::::::M     M:::::::::::M
//          A:::::A A:::::A            V:::::V       V:::::V   M:::::::M::::M   M::::M:::::::MM:::::::M::::M   M::::M:::::::M
//         A:::::A   A:::::A            V:::::V     V:::::V    M::::::M M::::M M::::M M::::::MM::::::M M::::M M::::M M::::::M
//        A:::::A     A:::::A            V:::::V   V:::::V     M::::::M  M::::M::::M  M::::::MM::::::M  M::::M::::M  M::::::M
//       A:::::AAAAAAAAA:::::A            V:::::V V:::::V      M::::::M   M:::::::M   M::::::MM::::::M   M:::::::M   M::::::M
//      A:::::::::::::::::::::A            V:::::V:::::V       M::::::M    M:::::M    M::::::MM::::::M    M:::::M    M::::::M
//     A:::::AAAAAAAAAAAAA:::::A            V:::::::::V        M::::::M     MMMMM     M::::::MM::::::M     MMMMM     M::::::M
//    A:::::A             A:::::A            V:::::::V         M::::::M               M::::::MM::::::M               M::::::M
//   A:::::A               A:::::A            V:::::V          M::::::M               M::::::MM::::::M               M::::::M
//  A:::::A                 A:::::A            V:::V           M::::::M               M::::::MM::::::M               M::::::M
// AAAAAAA                   AAAAAAA            VVV            MMMMMMMM               MMMMMMMMMMMMMMMM               MMMMMMMM

localparam DMA_N                        = 4;
localparam NO_AVMM_MASTERS              = 1;
localparam NO_BASE_ENGINES              = 1;
localparam NO_DBG_TAPS                  = 0;
localparam DBG_WIDTH                    = 2048;
localparam DDR_SIM                      = 0;

logic [NO_DBG_TAPS-1:0][DBG_WIDTH-1:0]  dbg_wires;

logic [NO_AVMM_MASTERS-1:0][1-1:0]      avmm_fh_read = 0;
logic [NO_AVMM_MASTERS-1:0][1-1:0]      avmm_fh_write = 0;
logic [NO_AVMM_MASTERS-1:0][32-1:0]     avmm_fh_address;
logic [NO_AVMM_MASTERS-1:0][32-1:0]     avmm_fh_writedata;
logic [NO_AVMM_MASTERS-1:0][32-1:0]     avmm_fh_readdata;
logic [NO_AVMM_MASTERS-1:0][32-1:0]     avmm_fh_readdata_r;
logic [NO_AVMM_MASTERS-1:0][1-1:0]      avmm_fh_readdatavalid;
logic [NO_AVMM_MASTERS-1:0][1-1:0]      avmm_fh_waitrequest;

logic [2-1:0] st_ocl;
logic [32-1:0] ocl_addr;

assign ocl_sh_arready = (st_ocl == 0);
assign ocl_sh_awready = (st_ocl == 0) & ~sh_ocl_arvalid;
assign ocl_sh_wready = (st_ocl == 1);
assign ocl_sh_rresp = '0;
assign ocl_sh_bresp = '0;

always_ff@(posedge clk_main_a0) begin
    integer i;
    for (i = 0; i < NO_AVMM_MASTERS; i ++) begin
        if (~avmm_fh_waitrequest[i]) begin
            avmm_fh_read                    [i] <= 0;
            avmm_fh_write                   [i] <= 0;
        end

        if (avmm_fh_readdatavalid[i])
            avmm_fh_readdata_r[i] <= avmm_fh_readdata[i];
    end

    case (st_ocl)
        0: begin
            if (sh_ocl_arvalid) begin
                ocl_addr                        <= sh_ocl_araddr;
                ocl_sh_rvalid                   <= 1;
                for (i = 0; i < NO_AVMM_MASTERS; i ++) begin
                    if (sh_ocl_araddr == (i*8)+0+`HELLO_WORLD_REG_ADDR)
                        ocl_sh_rdata            <= avmm_fh_readdata_r[i];
                end
                st_ocl                          <= 3;

            end else if (sh_ocl_awvalid) begin
                ocl_addr                        <= sh_ocl_awaddr;
                st_ocl                          <= 1;
            end
        end
        1: begin
            if (sh_ocl_wvalid) begin
                for (i = 0; i < NO_AVMM_MASTERS; i ++) begin
                    if (ocl_addr == (i*8)+0+`HELLO_WORLD_REG_ADDR) begin
                        avmm_fh_address     [i] <= sh_ocl_wdata[0+:20];
                        avmm_fh_read        [i] <= ~sh_ocl_wdata[32-1];
                        avmm_fh_write       [i] <= sh_ocl_wdata[32-1];
                    end
                    if (ocl_addr == (i*8)+4+`HELLO_WORLD_REG_ADDR) begin
                        avmm_fh_writedata   [i] <= sh_ocl_wdata;
                    end
                end
                ocl_sh_bvalid                   <= 1;
                st_ocl                          <= 2;
            end
        end
        2: begin
            if (sh_ocl_bready) begin
                ocl_sh_bvalid                   <= 0;

                st_ocl                          <= 0;
            end
        end

        3: begin
            if (sh_ocl_rready) begin
                ocl_sh_rvalid                   <= 0;
                st_ocl                          <= 0;
            end
        end

    endcase

    if (rst) begin
        st_ocl                                  <= 0;
        ocl_sh_rvalid                           <= '0;
        ocl_sh_bvalid                           <= '0;
        avmm_fh_read                            <= '0;
        avmm_fh_write                           <= '0;
    end
end

// DDDDDDDDDDDDD        DDDDDDDDDDDDD        RRRRRRRRRRRRRRRRR   
// D::::::::::::DDD     D::::::::::::DDD     R::::::::::::::::R  
// D:::::::::::::::DD   D:::::::::::::::DD   R::::::RRRRRR:::::R 
// DDD:::::DDDDD:::::D  DDD:::::DDDDD:::::D  RR:::::R     R:::::R
//   D:::::D    D:::::D   D:::::D    D:::::D   R::::R     R:::::R
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::RRRRRR:::::R 
//   D:::::D     D:::::D  D:::::D     D:::::D  R:::::::::::::RR  
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::RRRRRR:::::R 
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R
//   D:::::D    D:::::D   D:::::D    D:::::D   R::::R     R:::::R
// DDD:::::DDDDD:::::D  DDD:::::DDDDD:::::D  RR:::::R     R:::::R
// D:::::::::::::::DD   D:::::::::::::::DD   R::::::R     R:::::R
// D::::::::::::DDD     D::::::::::::DDD     R::::::R     R:::::R
// DDDDDDDDDDDDD        DDDDDDDDDDDDD        RRRRRRRR     RRRRRRR

logic [4-1:0][1-1:0]                        ddr_rd_en;
logic [4-1:0][1-1:0]                        ddr_rd_pop;
logic [4-1:0][64-1:0]                       ddr_rd_addr;
logic [4-1:0][9-1:0]                        ddr_rd_sz;
logic [4-1:0][1-1:0]                        ddr_rd_v;
logic [4-1:0][512-1:0]                      ddr_rd_data;

logic [4-1:0][1-1:0]                        ddr_wr_en;
logic [4-1:0][1-1:0]                        ddr_wr_pop;
logic [4-1:0][1-1:0]                        ddr_wr_res;
logic [4-1:0][64-1:0]                       ddr_wr_addr;
logic [4-1:0][512-1:0]                      ddr_wr_data;




logic [15:0]                                ddr_awid        [4-1:0];
logic [63:0]                                ddr_awaddr      [4-1:0];
logic [7:0]                                 ddr_awlen       [4-1:0];
logic [2:0]                                 ddr_awsize      [4-1:0];
logic [1:0]                                 ddr_awburst     [4-1:0];
logic [0:0]                                 ddr_awvalid     [4-1:0];
logic [4-1:0]                               ddr_awready;

logic [15:0]                                ddr_wid         [4-1:0];
logic [511:0]                               ddr_wdata       [4-1:0];
logic [63:0]                                ddr_wstrb       [4-1:0];
logic [4-1:0]                               ddr_wlast;
logic [4-1:0]                               ddr_wvalid;
logic [4-1:0]                               ddr_wready;

logic [15:0]                                ddr_bid         [4-1:0];
logic [1:0]                                 ddr_bresp       [4-1:0];
logic [4-1:0]                               ddr_bvalid;
logic [4-1:0]                               ddr_bready;

logic [15:0]                                ddr_arid        [4-1:0];
logic [63:0]                                ddr_araddr      [4-1:0];
logic [7:0]                                 ddr_arlen       [4-1:0];
logic [2:0]                                 ddr_arsize      [4-1:0];
logic [1:0]                                 ddr_arburst     [4-1:0];
logic [4-1:0]                               ddr_arvalid;
logic [4-1:0]                               ddr_arready;

logic [15:0]                                ddr_rid         [4-1:0];
logic [511:0]                               ddr_rdata       [4-1:0];
logic [1:0]                                 ddr_rresp       [4-1:0];
logic [4-1:0]                               ddr_rlast;
logic [4-1:0]                               ddr_rvalid;
logic [4-1:0]                               ddr_rready;

logic [4-1:0]                               ddr_is_ready;

generate
if (DDR_SIM == 0) begin

assign cl_sh_ddr_awid                   = ddr_awid[0];
assign cl_sh_ddr_awaddr                 = ddr_awaddr[0];
assign cl_sh_ddr_awlen                  = ddr_awlen[0];
assign cl_sh_ddr_awsize                 = ddr_awsize[0];
assign cl_sh_ddr_awburst                = ddr_awburst[0];
assign cl_sh_ddr_awvalid                = ddr_awvalid[0];
assign ddr_awready[0]                   = sh_cl_ddr_awready;

assign cl_sh_ddr_wid                    = ddr_wid[0];
assign cl_sh_ddr_wdata                  = ddr_wdata[0];
assign cl_sh_ddr_wstrb                  = ddr_wstrb[0];
assign cl_sh_ddr_wlast                  = ddr_wlast[0];
assign cl_sh_ddr_wvalid                 = ddr_wvalid[0];
assign ddr_wready[0]                    = sh_cl_ddr_wready;

assign ddr_bid[0]                       = sh_cl_ddr_bid;
assign ddr_bresp[0]                     = sh_cl_ddr_bresp;
assign ddr_bvalid[0]                    = sh_cl_ddr_bvalid;
assign cl_sh_ddr_bready                 = ddr_bready[0];

assign cl_sh_ddr_arid                   = ddr_arid[0];
assign cl_sh_ddr_araddr                 = ddr_araddr[0];
assign cl_sh_ddr_arlen                  = ddr_arlen[0];
assign cl_sh_ddr_arsize                 = ddr_arsize[0];
assign cl_sh_ddr_arburst                = ddr_arburst[0];
assign cl_sh_ddr_arvalid                = ddr_arvalid[0];
assign ddr_arready[0]                   = sh_cl_ddr_arready;

assign ddr_rid[0]                       = sh_cl_ddr_rid;
assign ddr_rdata[0]                     = sh_cl_ddr_rdata;
assign ddr_rresp[0]                     = sh_cl_ddr_rresp;
assign ddr_rlast[0]                     = sh_cl_ddr_rlast;
assign ddr_rvalid[0]                    = sh_cl_ddr_rvalid;
assign cl_sh_ddr_rready                 = ddr_rready[0];

assign ddr_is_ready[0]                  = sh_cl_ddr_is_ready;


logic [7:0] sh_ddr_stat_addr_q[2:0];
logic[2:0] sh_ddr_stat_wr_q;
logic[2:0] sh_ddr_stat_rd_q; 
logic[31:0] sh_ddr_stat_wdata_q[2:0];
logic[2:0] ddr_sh_stat_ack_q;
logic[31:0] ddr_sh_stat_rdata_q[2:0];
logic[7:0] ddr_sh_stat_int_q[2:0];


lib_pipe #(.WIDTH(1+1+8+32), .STAGES(NUM_CFG_STGS_CL_DDR_ATG)) PIPE_DDR_STAT0 (.clk(clk), .rst_n(sync_rst_n),
                                               .in_bus({sh_ddr_stat_wr0, sh_ddr_stat_rd0, sh_ddr_stat_addr0, sh_ddr_stat_wdata0}),
                                               .out_bus({sh_ddr_stat_wr_q[0], sh_ddr_stat_rd_q[0], sh_ddr_stat_addr_q[0], sh_ddr_stat_wdata_q[0]})
                                               );


lib_pipe #(.WIDTH(1+8+32), .STAGES(NUM_CFG_STGS_CL_DDR_ATG)) PIPE_DDR_STAT_ACK0 (.clk(clk), .rst_n(sync_rst_n),
                                               .in_bus({ddr_sh_stat_ack_q[0], ddr_sh_stat_int_q[0], ddr_sh_stat_rdata_q[0]}),
                                               .out_bus({ddr_sh_stat_ack0, ddr_sh_stat_int0, ddr_sh_stat_rdata0})
                                               );


lib_pipe #(.WIDTH(1+1+8+32), .STAGES(NUM_CFG_STGS_CL_DDR_ATG)) PIPE_DDR_STAT1 (.clk(clk), .rst_n(sync_rst_n),
                                               .in_bus({sh_ddr_stat_wr1, sh_ddr_stat_rd1, sh_ddr_stat_addr1, sh_ddr_stat_wdata1}),
                                               .out_bus({sh_ddr_stat_wr_q[1], sh_ddr_stat_rd_q[1], sh_ddr_stat_addr_q[1], sh_ddr_stat_wdata_q[1]})
                                               );


lib_pipe #(.WIDTH(1+8+32), .STAGES(NUM_CFG_STGS_CL_DDR_ATG)) PIPE_DDR_STAT_ACK1 (.clk(clk), .rst_n(sync_rst_n),
                                               .in_bus({ddr_sh_stat_ack_q[1], ddr_sh_stat_int_q[1], ddr_sh_stat_rdata_q[1]}),
                                               .out_bus({ddr_sh_stat_ack1, ddr_sh_stat_int1, ddr_sh_stat_rdata1})
                                               );

lib_pipe #(.WIDTH(1+1+8+32), .STAGES(NUM_CFG_STGS_CL_DDR_ATG)) PIPE_DDR_STAT2 (.clk(clk), .rst_n(sync_rst_n),
                                               .in_bus({sh_ddr_stat_wr2, sh_ddr_stat_rd2, sh_ddr_stat_addr2, sh_ddr_stat_wdata2}),
                                               .out_bus({sh_ddr_stat_wr_q[2], sh_ddr_stat_rd_q[2], sh_ddr_stat_addr_q[2], sh_ddr_stat_wdata_q[2]})
                                               );


lib_pipe #(.WIDTH(1+8+32), .STAGES(NUM_CFG_STGS_CL_DDR_ATG)) PIPE_DDR_STAT_ACK2 (.clk(clk), .rst_n(sync_rst_n),
                                               .in_bus({ddr_sh_stat_ack_q[2], ddr_sh_stat_int_q[2], ddr_sh_stat_rdata_q[2]}),
                                               .out_bus({ddr_sh_stat_ack2, ddr_sh_stat_int2, ddr_sh_stat_rdata2})
                                               ); 

(* dont_touch = "true" *) logic sh_ddr_sync_rst_n;
lib_pipe #(.WIDTH(1), .STAGES(4)) SH_DDR_SLC_RST_N (.clk(clk), .rst_n(1'b1), .in_bus(sync_rst_n), .out_bus(sh_ddr_sync_rst_n));
sh_ddr #(
         .DDR_A_PRESENT(`DDR_A_PRESENT),
         .DDR_B_PRESENT(`DDR_B_PRESENT),
         .DDR_D_PRESENT(0)
   ) SH_DDR
   (
   .clk(clk),
   .rst_n(sh_ddr_sync_rst_n),

   .stat_clk(clk),
   .stat_rst_n(sh_ddr_sync_rst_n),


   .CLK_300M_DIMM0_DP(CLK_300M_DIMM0_DP),
   .CLK_300M_DIMM0_DN(CLK_300M_DIMM0_DN),
   .M_A_ACT_N(M_A_ACT_N),
   .M_A_MA(M_A_MA),
   .M_A_BA(M_A_BA),
   .M_A_BG(M_A_BG),
   .M_A_CKE(M_A_CKE),
   .M_A_ODT(M_A_ODT),
   .M_A_CS_N(M_A_CS_N),
   .M_A_CLK_DN(M_A_CLK_DN),
   .M_A_CLK_DP(M_A_CLK_DP),
   .M_A_PAR(M_A_PAR),
   .M_A_DQ(M_A_DQ),
   .M_A_ECC(M_A_ECC),
   .M_A_DQS_DP(M_A_DQS_DP),
   .M_A_DQS_DN(M_A_DQS_DN),
   .cl_RST_DIMM_A_N(cl_RST_DIMM_A_N),
   
   
   .CLK_300M_DIMM1_DP(CLK_300M_DIMM1_DP),
   .CLK_300M_DIMM1_DN(CLK_300M_DIMM1_DN),
   .M_B_ACT_N(M_B_ACT_N),
   .M_B_MA(M_B_MA),
   .M_B_BA(M_B_BA),
   .M_B_BG(M_B_BG),
   .M_B_CKE(M_B_CKE),
   .M_B_ODT(M_B_ODT),
   .M_B_CS_N(M_B_CS_N),
   .M_B_CLK_DN(M_B_CLK_DN),
   .M_B_CLK_DP(M_B_CLK_DP),
   .M_B_PAR(M_B_PAR),
   .M_B_DQ(M_B_DQ),
   .M_B_ECC(M_B_ECC),
   .M_B_DQS_DP(M_B_DQS_DP),
   .M_B_DQS_DN(M_B_DQS_DN),
   .cl_RST_DIMM_B_N(cl_RST_DIMM_B_N),

   .CLK_300M_DIMM3_DP(CLK_300M_DIMM3_DP),
   .CLK_300M_DIMM3_DN(CLK_300M_DIMM3_DN),
   .M_D_ACT_N(M_D_ACT_N),
   .M_D_MA(M_D_MA),
   .M_D_BA(M_D_BA),
   .M_D_BG(M_D_BG),
   .M_D_CKE(M_D_CKE),
   .M_D_ODT(M_D_ODT),
   .M_D_CS_N(M_D_CS_N),
   .M_D_CLK_DN(M_D_CLK_DN),
   .M_D_CLK_DP(M_D_CLK_DP),
   .M_D_PAR(M_D_PAR),
   .M_D_DQ(M_D_DQ),
   .M_D_ECC(M_D_ECC),
   .M_D_DQS_DP(M_D_DQS_DP),
   .M_D_DQS_DN(M_D_DQS_DN),
   .cl_RST_DIMM_D_N(cl_RST_DIMM_D_N),

    //------------------------------------------------------
    // DDR-4 Interface from CL (AXI-4)
    //------------------------------------------------------
    .cl_sh_ddr_awid(ddr_awid[1+:3]),
    .cl_sh_ddr_awaddr(ddr_awaddr[1+:3]),
    .cl_sh_ddr_awlen(ddr_awlen[1+:3]),
    .cl_sh_ddr_awsize(ddr_awsize[1+:3]),
    .cl_sh_ddr_awvalid(ddr_awvalid[1+:3]),
    .cl_sh_ddr_awburst(ddr_awburst[1+:3]),
    .sh_cl_ddr_awready(ddr_awready[1+:3]),

    .cl_sh_ddr_wid(ddr_wid[1+:3]),
    .cl_sh_ddr_wdata(ddr_wdata[1+:3]),
    .cl_sh_ddr_wstrb(ddr_wstrb[1+:3]),
    .cl_sh_ddr_wlast(ddr_wlast[1+:3]),
    .cl_sh_ddr_wvalid(ddr_wvalid[1+:3]),
    .sh_cl_ddr_wready(ddr_wready[1+:3]),

    .sh_cl_ddr_bid(ddr_bid[1+:3]),
    .sh_cl_ddr_bresp(ddr_bresp[1+:3]),
    .sh_cl_ddr_bvalid(ddr_bvalid[1+:3]),
    .cl_sh_ddr_bready(ddr_bready[1+:3]),

    .cl_sh_ddr_arid(ddr_arid[1+:3]),
    .cl_sh_ddr_araddr(ddr_araddr[1+:3]),
    .cl_sh_ddr_arlen(ddr_arlen[1+:3]),
    .cl_sh_ddr_arsize(ddr_arsize[1+:3]),
    .cl_sh_ddr_arvalid(ddr_arvalid[1+:3]),
    .cl_sh_ddr_arburst(ddr_arburst[1+:3]),
    .sh_cl_ddr_arready(ddr_arready[1+:3]),

    .sh_cl_ddr_rid(ddr_rid[1+:3]),
    .sh_cl_ddr_rdata(ddr_rdata[1+:3]),
    .sh_cl_ddr_rresp(ddr_rresp[1+:3]),
    .sh_cl_ddr_rlast(ddr_rlast[1+:3]),
    .sh_cl_ddr_rvalid(ddr_rvalid[1+:3]),
    .cl_sh_ddr_rready(ddr_rready[1+:3]),

    .sh_cl_ddr_is_ready(ddr_is_ready[1+:3]),


   .sh_ddr_stat_addr0  (sh_ddr_stat_addr_q[0]) ,
   .sh_ddr_stat_wr0    (sh_ddr_stat_wr_q[0]     ) , 
   .sh_ddr_stat_rd0    (sh_ddr_stat_rd_q[0]     ) , 
   .sh_ddr_stat_wdata0 (sh_ddr_stat_wdata_q[0]  ) , 
   .ddr_sh_stat_ack0   (ddr_sh_stat_ack_q[0]    ) ,
   .ddr_sh_stat_rdata0 (ddr_sh_stat_rdata_q[0]  ),
   .ddr_sh_stat_int0   (ddr_sh_stat_int_q[0]    ),

   .sh_ddr_stat_addr1  (sh_ddr_stat_addr_q[1]) ,
   .sh_ddr_stat_wr1    (sh_ddr_stat_wr_q[1]     ) , 
   .sh_ddr_stat_rd1    (sh_ddr_stat_rd_q[1]     ) , 
   .sh_ddr_stat_wdata1 (sh_ddr_stat_wdata_q[1]  ) , 
   .ddr_sh_stat_ack1   (ddr_sh_stat_ack_q[1]    ) ,
   .ddr_sh_stat_rdata1 (ddr_sh_stat_rdata_q[1]  ),
   .ddr_sh_stat_int1   (ddr_sh_stat_int_q[1]    ),

   .sh_ddr_stat_addr2  (sh_ddr_stat_addr_q[2]) ,
   .sh_ddr_stat_wr2    (sh_ddr_stat_wr_q[2]     ) , 
   .sh_ddr_stat_rd2    (sh_ddr_stat_rd_q[2]     ) , 
   .sh_ddr_stat_wdata2 (sh_ddr_stat_wdata_q[2]  ) , 
   .ddr_sh_stat_ack2   (ddr_sh_stat_ack_q[2]    ) ,
   .ddr_sh_stat_rdata2 (ddr_sh_stat_rdata_q[2]  ),
   .ddr_sh_stat_int2   (ddr_sh_stat_int_q[2]    ) 
   );

end else begin

    for (genvar g_i = 0; g_i < 4; g_i ++) begin

        logic [512-1:0] mem [(1<<(26+3-6))-1:0];

        logic [1-1:0] raddr_v;
        logic [1-1:0] waddr_v;
        logic [1-1:0] wdata_v;
        logic [64-1:0] raddr_d;
        logic [8-1:0] raddr_s;
        logic [64-1:0] waddr_d;
        logic [512-1:0] wdata_d;

        logic [2-1:0] st_rd;
        logic [64-1:0] raddr_d_r;
        logic [8-1:0] raddr_s_r;

        assign ddr_is_ready[g_i] = 1;
        assign ddr_arready[g_i] = 1;
        assign ddr_awready[g_i] = 1;
        assign ddr_wready[g_i] = 1;

        showahead_fifo #(
            .WIDTH                              ($bits({ddr_arlen[g_i], ddr_araddr[g_i]})),
            .DEPTH                              (512)
        ) ddr_raddr_fifo_inst (
            .aclr                               (rst),

            .wr_clk                             (clk_main_a0),
            .wr_req                             (ddr_arvalid[g_i] & ddr_arready[g_i]),
            .wr_full                            (),
            .wr_data                            ({ddr_arlen[g_i], ddr_araddr[g_i]}),

            .rd_clk                             (clk_main_a0),
            .rd_req                             (raddr_v & (((st_rd == 0) & (raddr_s == 0)) | ((st_rd == 1) & (raddr_s_r == 0)))),
            .rd_empty                           (),
            .rd_not_empty                       (raddr_v),
            .rd_count                           (),
            .rd_data                            ({raddr_s, raddr_d})
        );
        showahead_fifo #(
            .WIDTH                              ($bits({ddr_awaddr[g_i]})),
            .DEPTH                              (512)
        ) ddr_waddr_fifo_inst (
            .aclr                               (rst),

            .wr_clk                             (clk_main_a0),
            .wr_req                             (ddr_awvalid[g_i] & ddr_awready[g_i]),
            .wr_full                            (),
            .wr_data                            ({ddr_awaddr[g_i]}),

            .rd_clk                             (clk_main_a0),
            .rd_req                             (waddr_v & wdata_v),
            .rd_empty                           (),
            .rd_not_empty                       (waddr_v),
            .rd_count                           (),
            .rd_data                            (waddr_d)
        );
        showahead_fifo #(
            .WIDTH                              ($bits({ddr_wdata[g_i]})),
            .DEPTH                              (512)
        ) ddr_wdata_fifo_inst (
            .aclr                               (rst),

            .wr_clk                             (clk_main_a0),
            .wr_req                             (ddr_wvalid[g_i] & ddr_wready[g_i]),
            .wr_full                            (),
            .wr_data                            ({ddr_wdata[g_i]}),

            .rd_clk                             (clk_main_a0),
            .rd_req                             (waddr_v & wdata_v),
            .rd_empty                           (),
            .rd_not_empty                       (wdata_v),
            .rd_count                           (),
            .rd_data                            (wdata_d)
        );

        always_ff@(posedge clk_main_a0) ddr_bvalid[g_i] <= waddr_v & wdata_v;
        always_ff@(posedge clk_main_a0) if (waddr_v & wdata_v) mem[waddr_d >> 6] <= wdata_d;
        always_ff@(posedge clk_main_a0) ddr_rvalid[g_i] <= raddr_v;

        always_ff@(posedge clk) begin
            case (st_rd)
                0: begin
                    raddr_s_r <= raddr_s - 1;
                    raddr_d_r <= raddr_d + (1<<6);
                    ddr_rdata[g_i] <= mem[raddr_d >> 6];
                    if (raddr_v & (raddr_s > 0))
                        st_rd <= 1;
                end
                1: begin
                    raddr_s_r <= raddr_s_r - 1;
                    raddr_d_r <= raddr_d_r + (1<<6);
                    ddr_rdata[g_i] <= mem[raddr_d_r >> 6];
                    if (raddr_s_r == 0)
                        st_rd <= 0;
                end
            endcase
            if (rst)
                st_rd <= 0;
        end

        always_ff@(posedge clk_main_a0)
        if (ddr_awvalid[g_i] & ddr_awready[g_i])
            $display("%t: %m.ddr_waddr[%0d]: %x",$time, g_i, ddr_awaddr[g_i]);
        always_ff@(posedge clk_main_a0)
        if (ddr_arvalid[g_i] & ddr_arready[g_i])
            $display("%t: %m.ddr_raddr[%0d]: %x",$time, g_i, ddr_araddr[g_i]);
        always_ff@(posedge clk_main_a0)
        if (ddr_wvalid[g_i] & ddr_wready[g_i])
            $display("%t: %m.ddr_wdata[%0d]: %x",$time, g_i, ddr_wdata[g_i]);
        always_ff@(posedge clk_main_a0)
        if (ddr_rvalid[g_i] & ddr_rready[g_i])
            $display("%t: %m.ddr_rdata[%0d]: %x",$time, g_i, ddr_rdata[g_i]);
        always_ff@(posedge clk_main_a0)
        if (ddr_bvalid[g_i] & ddr_bready[g_i])
            $display("%t: %m.ddr_wresp[%0d]: %x",$time, g_i, ddr_bresp[g_i]);

    end
end
endgenerate


generate
    for (genvar g_i = 0; g_i < 4; g_i ++) begin

        logic [2-1:0] ddr_w_st;

        always_comb begin
            ddr_awlen                                   [g_i]   = 0;
            ddr_awsize                                  [g_i]   = 6;
            ddr_awburst                                 [g_i]   = 2'b01;
            ddr_wstrb                                   [g_i]   = '1;
            ddr_wlast                                   [g_i]   = '1;
            ddr_bready                                  [g_i]   = '1;
            ddr_rready                                  [g_i]   = '1;

            ddr_arvalid                                 [g_i]   = ddr_rd_en     [g_i];
            ddr_araddr                                  [g_i]   = ddr_rd_addr   [g_i];
            ddr_arid                                    [g_i]   = 0;
            ddr_arlen                                   [g_i]   = ddr_rd_sz     [g_i] - 1;
            ddr_arsize                                  [g_i]   = 6;
            ddr_arburst                                 [g_i]   = 1;

            ddr_rd_pop                                  [g_i]   = ddr_rd_en     [g_i] & ddr_arready [g_i];
        end

        always_ff@(posedge clk) ddr_rd_v                [g_i]   <= ddr_rvalid    [g_i] & ddr_rready  [g_i];
        always_ff@(posedge clk) ddr_rd_data             [g_i]   <= ddr_rdata     [g_i];
        always_ff@(posedge clk) ddr_wr_res              [g_i]   <= ddr_bvalid    [g_i] & ddr_bready  [g_i];

        always_ff@(posedge clk) begin

            case (ddr_w_st)
                0: begin
                    ddr_awaddr                          [g_i]   <= ddr_wr_addr[g_i];
                    ddr_wdata                           [g_i]   <= ddr_wr_data[g_i];
                    ddr_awid                            [g_i]   <= 0;//ddr_awid                            [g_i]+1;
                    ddr_wid                             [g_i]   <= 0;//ddr_wid                             [g_i]+1;

                    if (ddr_wr_en[g_i]) begin
                        ddr_wr_pop                      [g_i]   <= 1;
                        ddr_awvalid                     [g_i]   <= 1;

                        ddr_wvalid                      [g_i]   <= 1;

                        ddr_w_st                                <= 1;
                    end
                end

                1: begin
                    ddr_wr_pop                          [g_i]   <= 0;

                    if (ddr_awready[g_i])
                        ddr_awvalid                     [g_i]   <= 0;

                    if (ddr_wready[g_i])
                        ddr_wvalid                      [g_i]   <= 0;

                    if ((ddr_awready[g_i] | ~ddr_awvalid[g_i]) & (ddr_wready[g_i] | ~ddr_wvalid[g_i])) begin
                        ddr_w_st                                <= 0;
                    end
                end
            endcase

            if (rst) begin
                ddr_wr_pop                              [g_i]   <= 0;
                ddr_awvalid                             [g_i]   <= 0;
                ddr_awid                                [g_i]   <= 0;
                ddr_wvalid                              [g_i]   <= 0;
                ddr_wid                                 [g_i]   <= 0;

                ddr_w_st                                        <= 0;
            end
        end
    end
endgenerate

















// PPPPPPPPPPPPPPPPP           CCCCCCCCCCCCCIIIIIIIIIIEEEEEEEEEEEEEEEEEEEEEE
// P::::::::::::::::P       CCC::::::::::::CI::::::::IE::::::::::::::::::::E
// P::::::PPPPPP:::::P    CC:::::::::::::::CI::::::::IE::::::::::::::::::::E
// PP:::::P     P:::::P  C:::::CCCCCCCC::::CII::::::IIEE::::::EEEEEEEEE::::E
//   P::::P     P:::::P C:::::C       CCCCCC  I::::I    E:::::E       EEEEEE
//   P::::P     P:::::PC:::::C                I::::I    E:::::E             
//   P::::PPPPPP:::::P C:::::C                I::::I    E::::::EEEEEEEEEE   
//   P:::::::::::::PP  C:::::C                I::::I    E:::::::::::::::E   
//   P::::PPPPPPPPP    C:::::C                I::::I    E:::::::::::::::E   
//   P::::P            C:::::C                I::::I    E::::::EEEEEEEEEE   
//   P::::P            C:::::C                I::::I    E:::::E             
//   P::::P             C:::::C       CCCCCC  I::::I    E:::::E       EEEEEE
// PP::::::PP            C:::::CCCCCCCC::::CII::::::IIEE::::::EEEEEEEE:::::E
// P::::::::P             CC:::::::::::::::CI::::::::IE::::::::::::::::::::E
// P::::::::P               CCC::::::::::::CI::::::::IE::::::::::::::::::::E
// PPPPPPPPPP                  CCCCCCCCCCCCCIIIIIIIIIIEEEEEEEEEEEEEEEEEEEEEE

logic [1-1:0] st_addr_v;
logic [3-1:0] st_data_v;
logic [2-1:0] st_v;
logic [1-1:0] st_p;
logic [64-1:0] st_addr;
logic [2-1:0][256-1:0] st_data;

assign cl_sh_dma_pcis_awready                   = 1'b1;
assign cl_sh_dma_pcis_wready                    = 1'b1;
assign cl_sh_dma_pcis_bresp                     = '0;

always_ff@(posedge clk) cl_sh_dma_pcis_bvalid   <= sh_cl_dma_pcis_wvalid & sh_cl_dma_pcis_wlast;
always_ff@(posedge clk) cl_sh_dma_pcis_bid      <= sh_cl_dma_pcis_awid;

showahead_fifo #(
    .WIDTH                              ($bits(sh_cl_dma_pcis_awaddr)),
    .DEPTH                              (32)
) st_in_addr_fifo_inst (
    .aclr                               (rst),

    .wr_clk                             (clk),
    .wr_req                             (sh_cl_dma_pcis_awvalid & cl_sh_dma_pcis_awready),
    .wr_full                            (),
    .wr_data                            ({sh_cl_dma_pcis_awaddr[64-1:6], 6'h0}),

    .rd_clk                             (clk),
    .rd_req                             (st_p),
    .rd_empty                           (),
    .rd_not_empty                       (st_addr_v),
    .rd_count                           (),
    .rd_data                            ({st_addr})
);

showahead_fifo #(
    .WIDTH                              ($bits({sh_cl_dma_pcis_wdata, sh_cl_dma_pcis_wstrb[32], sh_cl_dma_pcis_wstrb[0]})),
    .DEPTH                              (32)
) st_in_data_fifo_inst (
    .aclr                               (rst),

    .wr_clk                             (clk),
    .wr_req                             (sh_cl_dma_pcis_wvalid & cl_sh_dma_pcis_wready),
    .wr_full                            (),
    .wr_data                            ({sh_cl_dma_pcis_wdata, sh_cl_dma_pcis_wstrb[32], sh_cl_dma_pcis_wstrb[0]}),

    .rd_clk                             (clk),
    .rd_req                             (st_p),
    .rd_empty                           (),
    .rd_not_empty                       (st_data_v[2]),
    .rd_count                           (),
    .rd_data                            ({st_data, st_data_v[0+:2]})
);

assign st_p                             = st_addr_v & st_data_v[2];
assign st_v[0]                          = st_addr_v & st_data_v[2] & st_data_v[0];
assign st_v[1]                          = st_addr_v & st_data_v[2] & st_data_v[1];





                                                                                  
// DDDDDDDDDDDDD        MMMMMMMM               MMMMMMMM               AAA               
// D::::::::::::DDD     M:::::::M             M:::::::M              A:::A              
// D:::::::::::::::DD   M::::::::M           M::::::::M             A:::::A             
// DDD:::::DDDDD:::::D  M:::::::::M         M:::::::::M            A:::::::A            
//   D:::::D    D:::::D M::::::::::M       M::::::::::M           A:::::::::A           
//   D:::::D     D:::::DM:::::::::::M     M:::::::::::M          A:::::A:::::A          
//   D:::::D     D:::::DM:::::::M::::M   M::::M:::::::M         A:::::A A:::::A         
//   D:::::D     D:::::DM::::::M M::::M M::::M M::::::M        A:::::A   A:::::A        
//   D:::::D     D:::::DM::::::M  M::::M::::M  M::::::M       A:::::A     A:::::A       
//   D:::::D     D:::::DM::::::M   M:::::::M   M::::::M      A:::::AAAAAAAAA:::::A      
//   D:::::D     D:::::DM::::::M    M:::::M    M::::::M     A:::::::::::::::::::::A     
//   D:::::D    D:::::D M::::::M     MMMMM     M::::::M    A:::::AAAAAAAAAAAAA:::::A    
// DDD:::::DDDDD:::::D  M::::::M               M::::::M   A:::::A             A:::::A   
// D:::::::::::::::DD   M::::::M               M::::::M  A:::::A               A:::::A  
// D::::::::::::DDD     M::::::M               M::::::M A:::::A                 A:::::A 
// DDDDDDDDDDDDD        MMMMMMMM               MMMMMMMMAAAAAAA                   AAAAAAA


logic [1-1:0]                           dma_push;
logic [DMA_N-1:0][14*32-1:0]            dma_push_d;
logic [1-1:0]                           dma_full_a;
logic [1-1:0]                           dma_full_d;

logic [14-1:0][32-1:0]                  dma_d;
logic [64-1:0]                          dma_addr;
logic [32-1:0]                          dma_seq_a;
logic [32-1:0]                          dma_seq_d;

logic [512/32-1:0][32-1:0]              cl_sh_pcim_wdata_w;

assign cl_sh_pcim_awid                  = '0;
assign cl_sh_pcim_awlen                 = '0;
assign cl_sh_pcim_awsize                = 'b110;
assign cl_sh_pcim_wstrb                 = '1;
assign cl_sh_pcim_wlast                 = '1;
assign cl_sh_pcim_bready                = '1;
assign cl_sh_pcim_wdata                 = cl_sh_pcim_wdata_w;

showahead_fifo_nx1 #(
    .N                                  (DMA_N),
    .WIDTH                              (1),
    .FULL_THRESH                        (512-64),
    .DEPTH                              (512)
) dma_addr_fifo_inst (
    .aclr                               (rst),

    .wr_clk                             (clk),
    .wr_req                             (dma_push),
    .wr_full                            (dma_full_a),
    .wr_data                            ('0),

    .rd_clk                             (clk),
    .rd_req                             (cl_sh_pcim_awvalid & sh_cl_pcim_awready),
    .rd_all                             (0),
    .rd_empty                           (),
    .rd_not_empty                       (cl_sh_pcim_awvalid),
    .rd_count                           (),
    .rd_data                            (),
    .rd_data_all                        ()
);

showahead_fifo_nx1 #(
    .N                                  (DMA_N),
    .WIDTH                              ($bits(dma_push_d)/DMA_N),
    .FULL_THRESH                        (512-64),
    .DEPTH                              (512)
) dma_data_fifo_inst (
    .aclr                               (rst),

    .wr_clk                             (clk),
    .wr_req                             (dma_push),
    .wr_full                            (dma_full_d),
    .wr_data                            ({dma_push_d}),

    .rd_clk                             (clk),
    .rd_req                             (cl_sh_pcim_wvalid & sh_cl_pcim_wready),
    .rd_all                             (0),
    .rd_empty                           (),
    .rd_not_empty                       (cl_sh_pcim_wvalid),
    .rd_count                           (),
    .rd_data                            (dma_d),
    .rd_data_all                        ()
);

always_comb begin
    integer i;

    // lower 12 bits (4kbyte pages) are offset
    cl_sh_pcim_awaddr                   = {dma_addr[64-1:12], dma_seq_a[0+:6], 6'h0};

    cl_sh_pcim_wdata_w[0]               = dma_seq_d;
    cl_sh_pcim_wdata_w[8]               = dma_seq_d;
    for (i = 0; i < 8-1; i ++) begin
        cl_sh_pcim_wdata_w[0+1+i]       = dma_d[i];
        cl_sh_pcim_wdata_w[8+1+i]       = dma_d[i+8-1];
    end
end

always_ff@(posedge clk) begin

    dma_seq_a                           <= dma_seq_a + (cl_sh_pcim_awvalid & sh_cl_pcim_awready);
    dma_seq_d                           <= dma_seq_d + (cl_sh_pcim_wvalid & sh_cl_pcim_wready);

    if (rst) begin
        dma_seq_a                       <= 32'h0ACE_0000;
        dma_seq_d                       <= 32'h0ACE_0000;
    end
end











// TTTTTTTTTTTTTTTTTTTTTTT               AAA               PPPPPPPPPPPPPPPPP   
// T:::::::::::::::::::::T              A:::A              P::::::::::::::::P  
// T:::::::::::::::::::::T             A:::::A             P::::::PPPPPP:::::P 
// T:::::TT:::::::TT:::::T            A:::::::A            PP:::::P     P:::::P
// TTTTTT  T:::::T  TTTTTT           A:::::::::A             P::::P     P:::::P
//         T:::::T                  A:::::A:::::A            P::::P     P:::::P
//         T:::::T                 A:::::A A:::::A           P::::PPPPPP:::::P 
//         T:::::T                A:::::A   A:::::A          P:::::::::::::PP  
//         T:::::T               A:::::A     A:::::A         P::::PPPPPPPPP    
//         T:::::T              A:::::AAAAAAAAA:::::A        P::::P            
//         T:::::T             A:::::::::::::::::::::A       P::::P            
//         T:::::T            A:::::AAAAAAAAAAAAA:::::A      P::::P            
//       TT:::::::TT         A:::::A             A:::::A   PP::::::PP          
//       T:::::::::T        A:::::A               A:::::A  P::::::::P          
//       T:::::::::T       A:::::A                 A:::::A P::::::::P          
//       TTTTTTTTTTT      AAAAAAA                   AAAAAAAPPPPPPPPPP          


generate
    if (NO_DBG_TAPS > 0) assign dbg_wires[0] = {

/*    input wire [64-1:0]                                */ sh_cl_dma_pcis_awaddr,
/*    input wire [6-1:0]                                 */ sh_cl_dma_pcis_awid,
/*    input wire [8-1:0]                                 */ sh_cl_dma_pcis_awlen,
/*    input wire [3-1:0]                                 */ sh_cl_dma_pcis_awsize,
/*    input wire [1-1:0]                                 */ sh_cl_dma_pcis_awvalid,
/*    output logic [1-1:0]                               */ cl_sh_dma_pcis_awready,

/*    input wire [64-1:0]                                */ sh_cl_dma_pcis_wstrb,
/*    input wire [1-1:0]                                 */ sh_cl_dma_pcis_wlast,
/*    input wire [1-1:0]                                 */ sh_cl_dma_pcis_wvalid,
/*    input wire [512-1:0]                               */ sh_cl_dma_pcis_wdata[0+:64],
/*    output logic [1-1:0]                               */ cl_sh_dma_pcis_wready,

/*    output logic [1:0]                                 */ cl_sh_dma_pcis_bresp,
/*    output logic [6-1:0]                               */ cl_sh_dma_pcis_bid,
/*    output logic [1-1:0]                               */ cl_sh_dma_pcis_bvalid,
/*    input wire [1-1:0]                                 */ sh_cl_dma_pcis_bready,



/*    output logic [16-1:0]                              */ cl_sh_pcim_awid,
/*    output logic [64-1:0]                              */ cl_sh_pcim_awaddr,
/*    output logic [8-1:0]                               */ cl_sh_pcim_awlen,
/*    output logic [3-1:0]                               */ cl_sh_pcim_awsize,
/*    output logic [19-1:0]                              */ cl_sh_pcim_awuser,
/*    output logic                                       */ cl_sh_pcim_awvalid,
/*    input wire [1-1:0]                                 */ sh_cl_pcim_awready,

/*    output logic [64-1:0]                              */ cl_sh_pcim_wstrb,
/*    output logic [1-1:0]                               */ cl_sh_pcim_wlast,
/*    output logic [1-1:0]                               */ cl_sh_pcim_wvalid,
/*    output logic [512-1:0]                             */ cl_sh_pcim_wdata[0+:64],
/*    input wire [1-1:0]                                 */ sh_cl_pcim_wready,

/*    input wire  [16-1:0]                               */ sh_cl_pcim_bid,
/*    input wire  [2-1:0]                                */ sh_cl_pcim_bresp,
/*    input wire [1-1:0]                                 */ sh_cl_pcim_bvalid,
/*    output logic [1-1:0]                               */ cl_sh_pcim_bready,

        ddr_rdata    [0][0+:64],ddr_rdata    [1][0+:64],ddr_rdata    [2][0+:64],
        ddr_wdata    [0][0+:64],ddr_wdata    [1][0+:64],ddr_wdata    [2][0+:64],

        ddr_awid            [0],ddr_awid            [1],ddr_awid            [2],
        ddr_awaddr          [0],ddr_awaddr          [1],ddr_awaddr          [2],
        ddr_awlen           [0],ddr_awlen           [1],ddr_awlen           [2],
        ddr_awsize          [0],ddr_awsize          [1],ddr_awsize          [2],
        ddr_awburst         [0],ddr_awburst         [1],ddr_awburst         [2],
        ddr_awvalid         [0],ddr_awvalid         [1],ddr_awvalid         [2],
        ddr_awready         [0],ddr_awready         [1],ddr_awready         [2],

        ddr_wid             [0],ddr_wid             [1],ddr_wid             [2],
        ddr_wlast           [0],ddr_wlast           [1],ddr_wlast           [2],
        ddr_wvalid          [0],ddr_wvalid          [1],ddr_wvalid          [2],
        ddr_wready          [0],ddr_wready          [1],ddr_wready          [2],

        ddr_bid             [0],ddr_bid             [1],ddr_bid             [2],
        ddr_bresp           [0],ddr_bresp           [1],ddr_bresp           [2],
        ddr_bvalid          [0],ddr_bvalid          [1],ddr_bvalid          [2],
        ddr_bready          [0],ddr_bready          [1],ddr_bready          [2],

        ddr_arid            [0],ddr_arid            [1],ddr_arid            [2],
        ddr_araddr          [0],ddr_araddr          [1],ddr_araddr          [2],
        ddr_arlen           [0],ddr_arlen           [1],ddr_arlen           [2],
        ddr_arsize          [0],ddr_arsize          [1],ddr_arsize          [2],
        ddr_arburst         [0],ddr_arburst         [1],ddr_arburst         [2],
        ddr_arvalid         [0],ddr_arvalid         [1],ddr_arvalid         [2],
        ddr_arready         [0],ddr_arready         [1],ddr_arready         [2],

        ddr_rid             [0],ddr_rid             [1],ddr_rid             [2],
        ddr_rresp           [0],ddr_rresp           [1],ddr_rresp           [2],
        ddr_rlast           [0],ddr_rlast           [1],ddr_rlast           [2],
        ddr_rvalid          [0],ddr_rvalid          [1],ddr_rvalid          [2],
        ddr_rready          [0],ddr_rready          [1],ddr_rready          [2],

        ddr_is_ready        [0],ddr_is_ready        [1],ddr_is_ready        [2],

        |{
            sh_cl_dma_pcis_awvalid,
            cl_sh_pcim_awvalid,
            ddr_awvalid[0],ddr_awvalid[1],ddr_awvalid[2],
            ddr_arvalid[0],ddr_arvalid[1],ddr_arvalid[2],
            ddr_wvalid[0],ddr_wvalid[1],ddr_wvalid[2],
            ddr_rvalid[0],ddr_rvalid[1],ddr_rvalid[2]
        },
        rst,
        clk
    };

    for (genvar g_d = 0; g_d < NO_DBG_TAPS; g_d ++) begin: G_D
        jtap #(
            .JTAP_ID                            (32'hACE1_0050 + g_d),
            .CAPTURE_LEN                        (1024),
            .NO_TRIGGERS                        (1+1),
            .DATA_W                             (DBG_WIDTH-3),
            .REGISTER_TAP                       (1)
        ) dbg_tap_inst (
            .tap_clk                            (dbg_wires[g_d][0]),
            .tap_reset                          (dbg_wires[g_d][1]),
            .triggers                           ({1'b1, dbg_wires[g_d][2]}),
            .tap_data                           (dbg_wires[g_d][DBG_WIDTH-1:3]),
            .avmm_clk                           (clk),
            .avmm_reset                         (rst),
            .avmm_read                          (avmm_fh_read            [NO_BASE_ENGINES+g_d]),
            .avmm_write                         (avmm_fh_write           [NO_BASE_ENGINES+g_d]),
            .avmm_address                       (avmm_fh_address         [NO_BASE_ENGINES+g_d]),
            .avmm_writedata                     (avmm_fh_writedata       [NO_BASE_ENGINES+g_d]),
            .avmm_readdata                      (avmm_fh_readdata        [NO_BASE_ENGINES+g_d]),
            .avmm_readdatavalid                 (avmm_fh_readdatavalid   [NO_BASE_ENGINES+g_d]),
            .avmm_waitrequest                   (avmm_fh_waitrequest     [NO_BASE_ENGINES+g_d])
        );
    end
endgenerate












//                AAA               PPPPPPPPPPPPPPPPP   PPPPPPPPPPPPPPPPP   
//               A:::A              P::::::::::::::::P  P::::::::::::::::P  
//              A:::::A             P::::::PPPPPP:::::P P::::::PPPPPP:::::P 
//             A:::::::A            PP:::::P     P:::::PPP:::::P     P:::::P
//            A:::::::::A             P::::P     P:::::P  P::::P     P:::::P
//           A:::::A:::::A            P::::P     P:::::P  P::::P     P:::::P
//          A:::::A A:::::A           P::::PPPPPP:::::P   P::::PPPPPP:::::P 
//         A:::::A   A:::::A          P:::::::::::::PP    P:::::::::::::PP  
//        A:::::A     A:::::A         P::::PPPPPPPPP      P::::PPPPPPPPP    
//       A:::::AAAAAAAAA:::::A        P::::P              P::::P            
//      A:::::::::::::::::::::A       P::::P              P::::P            
//     A:::::AAAAAAAAAAAAA:::::A      P::::P              P::::P            
//    A:::::A             A:::::A   PP::::::PP          PP::::::PP          
//   A:::::A               A:::::A  P::::::::P          P::::::::P          
//  A:::::A                 A:::::A P::::::::P          P::::::::P          
// AAAAAAA                   AAAAAAAPPPPPPPPPP          PPPPPPPPPP          

`ifndef TOP_NAME
`define TOP_NAME top_msm_2
`endif

`TOP_NAME #(
    .DBG_WIDTH(DBG_WIDTH),
    .DMA_N                                              (DMA_N)
) top_inst (

    .avmm_read                                          (avmm_fh_read[0]),
    .avmm_write                                         (avmm_fh_write[0]),
    .avmm_address                                       (avmm_fh_address[0]),
    .avmm_writedata                                     (avmm_fh_writedata[0]),
    .avmm_readdata                                      (avmm_fh_readdata[0]),
    .avmm_readdatavalid                                 (avmm_fh_readdatavalid[0]),
    .avmm_waitrequest                                   (avmm_fh_waitrequest[0]),

    .pcie_v                                             (st_v),
    .pcie_a                                             (st_addr),
    .pcie_d                                             (st_data),

    .dma_v                                              (dma_push),
    .dma_a                                              (dma_addr),
    .dma_f                                              (dma_full_a | dma_full_d),
    .dma_d                                              (dma_push_d),
    .dma_s                                              (dma_seq_a),

    .ddr_rd_en                                          (ddr_rd_en),
    .ddr_rd_pop                                         (ddr_rd_pop),
    .ddr_rd_addr                                        (ddr_rd_addr),
    .ddr_rd_sz                                          (ddr_rd_sz),
    .ddr_rd_v                                           (ddr_rd_v),
    .ddr_rd_data                                        (ddr_rd_data),

    .ddr_wr_en                                          (ddr_wr_en),
    .ddr_wr_pop                                         (ddr_wr_pop),
    .ddr_wr_res                                         (ddr_wr_res),
    .ddr_wr_addr                                        (ddr_wr_addr),
    .ddr_wr_data                                        (ddr_wr_data),

    .dbg_wire                                           (dbg_wires[1]),

    .clk(clk),
    .rst(rst)
);

always_ff@(posedge clk)
if (sh_cl_dma_pcis_awvalid)
$display("%t: %m.pcis_addr: %x %x %x",$time, sh_cl_dma_pcis_awaddr, sh_cl_dma_pcis_awlen, sh_cl_dma_pcis_awsize);
always_ff@(posedge clk)
if (sh_cl_dma_pcis_wvalid)
$display("%t: %m.pcis_data: %x %x",$time, sh_cl_dma_pcis_wdata, sh_cl_dma_pcis_wstrb);


always_ff@(negedge clk)
$display("%t: -------------",$time);

endmodule   
