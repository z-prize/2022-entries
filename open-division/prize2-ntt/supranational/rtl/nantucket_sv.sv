// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Breaking naming conventions here a bit to speed up
// coding.  Use names that match port names so the
// port wildcarding works.

module nantucket_sv
  #(
    parameter int C_S_AXI_CONTROL_ADDR_WIDTH = 12 ,
    parameter int C_S_AXI_CONTROL_DATA_WIDTH = 32 ,
    parameter int C_M00_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M00_AXI_DATA_WIDTH       = 256,
    parameter int C_M01_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M01_AXI_DATA_WIDTH       = 256,
    parameter int C_M02_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M02_AXI_DATA_WIDTH       = 256,
    parameter int C_M03_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M03_AXI_DATA_WIDTH       = 256,
    parameter int C_M04_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M04_AXI_DATA_WIDTH       = 256,
    parameter int C_M05_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M05_AXI_DATA_WIDTH       = 256,
    parameter int C_M06_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M06_AXI_DATA_WIDTH       = 256,
    parameter int C_M07_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M07_AXI_DATA_WIDTH       = 256,
    parameter int C_M08_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M08_AXI_DATA_WIDTH       = 256,
    parameter int C_M09_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M09_AXI_DATA_WIDTH       = 256,
    parameter int C_M10_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M10_AXI_DATA_WIDTH       = 256,
    parameter int C_M11_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M11_AXI_DATA_WIDTH       = 256,
    parameter int C_M12_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M12_AXI_DATA_WIDTH       = 256,
    parameter int C_M13_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M13_AXI_DATA_WIDTH       = 256,
    parameter int C_M14_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M14_AXI_DATA_WIDTH       = 256,
    parameter int C_M15_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M15_AXI_DATA_WIDTH       = 256,
    parameter int C_M16_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M16_AXI_DATA_WIDTH       = 256,
    parameter int C_M17_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M17_AXI_DATA_WIDTH       = 256,
    parameter int C_M18_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M18_AXI_DATA_WIDTH       = 256,
    parameter int C_M19_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M19_AXI_DATA_WIDTH       = 256,
    parameter int C_M20_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M20_AXI_DATA_WIDTH       = 256,
    parameter int C_M21_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M21_AXI_DATA_WIDTH       = 256,
    parameter int C_M22_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M22_AXI_DATA_WIDTH       = 256,
    parameter int C_M23_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M23_AXI_DATA_WIDTH       = 256,
    parameter int C_M24_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M24_AXI_DATA_WIDTH       = 256,
    parameter int C_M25_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M25_AXI_DATA_WIDTH       = 256,
    parameter int C_M26_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M26_AXI_DATA_WIDTH       = 256,
    parameter int C_M27_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M27_AXI_DATA_WIDTH       = 256,
    parameter int C_M28_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M28_AXI_DATA_WIDTH       = 256,
    parameter int C_M29_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M29_AXI_DATA_WIDTH       = 256,
    parameter int C_M30_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M30_AXI_DATA_WIDTH       = 256,
    parameter int C_M31_AXI_ADDR_WIDTH       = 64 ,
    parameter int C_M31_AXI_DATA_WIDTH       = 256
    )
  (
   input  logic dma_clk_i,
   input  logic dma_rst_ni,
   input  logic ntt_clk_i,

   // Tool generated ports.
   output logic                              m00_axi_awvalid,
   input  logic                              m00_axi_awready,
   output logic [C_M00_AXI_ADDR_WIDTH-1:0]   m00_axi_awaddr ,
   output logic [8-1:0]                      m00_axi_awlen  ,
   output logic                              m00_axi_wvalid ,
   input  logic                              m00_axi_wready ,
   output logic [C_M00_AXI_DATA_WIDTH-1:0]   m00_axi_wdata  ,
   output logic [C_M00_AXI_DATA_WIDTH/8-1:0] m00_axi_wstrb  ,
   output logic                              m00_axi_wlast  ,
   input  logic                              m00_axi_bvalid ,
   output logic                              m00_axi_bready ,
   output logic                              m00_axi_arvalid,
   input  logic                              m00_axi_arready,
   output logic [C_M00_AXI_ADDR_WIDTH-1:0]   m00_axi_araddr ,
   output logic [8-1:0]                      m00_axi_arlen  ,
   input  logic                              m00_axi_rvalid ,
   output logic                              m00_axi_rready ,
   input  logic [C_M00_AXI_DATA_WIDTH-1:0]   m00_axi_rdata  ,
   input  logic                              m00_axi_rlast  ,

   output logic                              m01_axi_awvalid,
   input  logic                              m01_axi_awready,
   output logic [C_M01_AXI_ADDR_WIDTH-1:0]   m01_axi_awaddr ,
   output logic [8-1:0]                      m01_axi_awlen  ,
   output logic                              m01_axi_wvalid ,
   input  logic                              m01_axi_wready ,
   output logic [C_M01_AXI_DATA_WIDTH-1:0]   m01_axi_wdata  ,
   output logic [C_M01_AXI_DATA_WIDTH/8-1:0] m01_axi_wstrb  ,
   output logic                              m01_axi_wlast  ,
   input  logic                              m01_axi_bvalid ,
   output logic                              m01_axi_bready ,
   output logic                              m01_axi_arvalid,
   input  logic                              m01_axi_arready,
   output logic [C_M01_AXI_ADDR_WIDTH-1:0]   m01_axi_araddr ,
   output logic [8-1:0]                      m01_axi_arlen  ,
   input  logic                              m01_axi_rvalid ,
   output logic                              m01_axi_rready ,
   input  logic [C_M01_AXI_DATA_WIDTH-1:0]   m01_axi_rdata  ,
   input  logic                              m01_axi_rlast  ,

   output logic                              m02_axi_awvalid,
   input  logic                              m02_axi_awready,
   output logic [C_M02_AXI_ADDR_WIDTH-1:0]   m02_axi_awaddr ,
   output logic [8-1:0]                      m02_axi_awlen  ,
   output logic                              m02_axi_wvalid ,
   input  logic                              m02_axi_wready ,
   output logic [C_M02_AXI_DATA_WIDTH-1:0]   m02_axi_wdata  ,
   output logic [C_M02_AXI_DATA_WIDTH/8-1:0] m02_axi_wstrb  ,
   output logic                              m02_axi_wlast  ,
   input  logic                              m02_axi_bvalid ,
   output logic                              m02_axi_bready ,
   output logic                              m02_axi_arvalid,
   input  logic                              m02_axi_arready,
   output logic [C_M02_AXI_ADDR_WIDTH-1:0]   m02_axi_araddr ,
   output logic [8-1:0]                      m02_axi_arlen  ,
   input  logic                              m02_axi_rvalid ,
   output logic                              m02_axi_rready ,
   input  logic [C_M02_AXI_DATA_WIDTH-1:0]   m02_axi_rdata  ,
   input  logic                              m02_axi_rlast  ,

   output logic                              m03_axi_awvalid,
   input  logic                              m03_axi_awready,
   output logic [C_M03_AXI_ADDR_WIDTH-1:0]   m03_axi_awaddr ,
   output logic [8-1:0]                      m03_axi_awlen  ,
   output logic                              m03_axi_wvalid ,
   input  logic                              m03_axi_wready ,
   output logic [C_M03_AXI_DATA_WIDTH-1:0]   m03_axi_wdata  ,
   output logic [C_M03_AXI_DATA_WIDTH/8-1:0] m03_axi_wstrb  ,
   output logic                              m03_axi_wlast  ,
   input  logic                              m03_axi_bvalid ,
   output logic                              m03_axi_bready ,
   output logic                              m03_axi_arvalid,
   input  logic                              m03_axi_arready,
   output logic [C_M03_AXI_ADDR_WIDTH-1:0]   m03_axi_araddr ,
   output logic [8-1:0]                      m03_axi_arlen  ,
   input  logic                              m03_axi_rvalid ,
   output logic                              m03_axi_rready ,
   input  logic [C_M03_AXI_DATA_WIDTH-1:0]   m03_axi_rdata  ,
   input  logic                              m03_axi_rlast  ,

   output logic                              m04_axi_awvalid,
   input  logic                              m04_axi_awready,
   output logic [C_M04_AXI_ADDR_WIDTH-1:0]   m04_axi_awaddr ,
   output logic [8-1:0]                      m04_axi_awlen  ,
   output logic                              m04_axi_wvalid ,
   input  logic                              m04_axi_wready ,
   output logic [C_M04_AXI_DATA_WIDTH-1:0]   m04_axi_wdata  ,
   output logic [C_M04_AXI_DATA_WIDTH/8-1:0] m04_axi_wstrb  ,
   output logic                              m04_axi_wlast  ,
   input  logic                              m04_axi_bvalid ,
   output logic                              m04_axi_bready ,
   output logic                              m04_axi_arvalid,
   input  logic                              m04_axi_arready,
   output logic [C_M04_AXI_ADDR_WIDTH-1:0]   m04_axi_araddr ,
   output logic [8-1:0]                      m04_axi_arlen  ,
   input  logic                              m04_axi_rvalid ,
   output logic                              m04_axi_rready ,
   input  logic [C_M04_AXI_DATA_WIDTH-1:0]   m04_axi_rdata  ,
   input  logic                              m04_axi_rlast  ,

   output logic                              m05_axi_awvalid,
   input  logic                              m05_axi_awready,
   output logic [C_M05_AXI_ADDR_WIDTH-1:0]   m05_axi_awaddr ,
   output logic [8-1:0]                      m05_axi_awlen  ,
   output logic                              m05_axi_wvalid ,
   input  logic                              m05_axi_wready ,
   output logic [C_M05_AXI_DATA_WIDTH-1:0]   m05_axi_wdata  ,
   output logic [C_M05_AXI_DATA_WIDTH/8-1:0] m05_axi_wstrb  ,
   output logic                              m05_axi_wlast  ,
   input  logic                              m05_axi_bvalid ,
   output logic                              m05_axi_bready ,
   output logic                              m05_axi_arvalid,
   input  logic                              m05_axi_arready,
   output logic [C_M05_AXI_ADDR_WIDTH-1:0]   m05_axi_araddr ,
   output logic [8-1:0]                      m05_axi_arlen  ,
   input  logic                              m05_axi_rvalid ,
   output logic                              m05_axi_rready ,
   input  logic [C_M05_AXI_DATA_WIDTH-1:0]   m05_axi_rdata  ,
   input  logic                              m05_axi_rlast  ,

   output logic                              m06_axi_awvalid,
   input  logic                              m06_axi_awready,
   output logic [C_M06_AXI_ADDR_WIDTH-1:0]   m06_axi_awaddr ,
   output logic [8-1:0]                      m06_axi_awlen  ,
   output logic                              m06_axi_wvalid ,
   input  logic                              m06_axi_wready ,
   output logic [C_M06_AXI_DATA_WIDTH-1:0]   m06_axi_wdata  ,
   output logic [C_M06_AXI_DATA_WIDTH/8-1:0] m06_axi_wstrb  ,
   output logic                              m06_axi_wlast  ,
   input  logic                              m06_axi_bvalid ,
   output logic                              m06_axi_bready ,
   output logic                              m06_axi_arvalid,
   input  logic                              m06_axi_arready,
   output logic [C_M06_AXI_ADDR_WIDTH-1:0]   m06_axi_araddr ,
   output logic [8-1:0]                      m06_axi_arlen  ,
   input  logic                              m06_axi_rvalid ,
   output logic                              m06_axi_rready ,
   input  logic [C_M06_AXI_DATA_WIDTH-1:0]   m06_axi_rdata  ,
   input  logic                              m06_axi_rlast  ,

   output logic                              m07_axi_awvalid,
   input  logic                              m07_axi_awready,
   output logic [C_M07_AXI_ADDR_WIDTH-1:0]   m07_axi_awaddr ,
   output logic [8-1:0]                      m07_axi_awlen  ,
   output logic                              m07_axi_wvalid ,
   input  logic                              m07_axi_wready ,
   output logic [C_M07_AXI_DATA_WIDTH-1:0]   m07_axi_wdata  ,
   output logic [C_M07_AXI_DATA_WIDTH/8-1:0] m07_axi_wstrb  ,
   output logic                              m07_axi_wlast  ,
   input  logic                              m07_axi_bvalid ,
   output logic                              m07_axi_bready ,
   output logic                              m07_axi_arvalid,
   input  logic                              m07_axi_arready,
   output logic [C_M07_AXI_ADDR_WIDTH-1:0]   m07_axi_araddr ,
   output logic [8-1:0]                      m07_axi_arlen  ,
   input  logic                              m07_axi_rvalid ,
   output logic                              m07_axi_rready ,
   input  logic [C_M07_AXI_DATA_WIDTH-1:0]   m07_axi_rdata  ,
   input  logic                              m07_axi_rlast  ,

   output logic                              m08_axi_awvalid,
   input  logic                              m08_axi_awready,
   output logic [C_M08_AXI_ADDR_WIDTH-1:0]   m08_axi_awaddr ,
   output logic [8-1:0]                      m08_axi_awlen  ,
   output logic                              m08_axi_wvalid ,
   input  logic                              m08_axi_wready ,
   output logic [C_M08_AXI_DATA_WIDTH-1:0]   m08_axi_wdata  ,
   output logic [C_M08_AXI_DATA_WIDTH/8-1:0] m08_axi_wstrb  ,
   output logic                              m08_axi_wlast  ,
   input  logic                              m08_axi_bvalid ,
   output logic                              m08_axi_bready ,
   output logic                              m08_axi_arvalid,
   input  logic                              m08_axi_arready,
   output logic [C_M08_AXI_ADDR_WIDTH-1:0]   m08_axi_araddr ,
   output logic [8-1:0]                      m08_axi_arlen  ,
   input  logic                              m08_axi_rvalid ,
   output logic                              m08_axi_rready ,
   input  logic [C_M08_AXI_DATA_WIDTH-1:0]   m08_axi_rdata  ,
   input  logic                              m08_axi_rlast  ,

   output logic                              m09_axi_awvalid,
   input  logic                              m09_axi_awready,
   output logic [C_M09_AXI_ADDR_WIDTH-1:0]   m09_axi_awaddr ,
   output logic [8-1:0]                      m09_axi_awlen  ,
   output logic                              m09_axi_wvalid ,
   input  logic                              m09_axi_wready ,
   output logic [C_M09_AXI_DATA_WIDTH-1:0]   m09_axi_wdata  ,
   output logic [C_M09_AXI_DATA_WIDTH/8-1:0] m09_axi_wstrb  ,
   output logic                              m09_axi_wlast  ,
   input  logic                              m09_axi_bvalid ,
   output logic                              m09_axi_bready ,
   output logic                              m09_axi_arvalid,
   input  logic                              m09_axi_arready,
   output logic [C_M09_AXI_ADDR_WIDTH-1:0]   m09_axi_araddr ,
   output logic [8-1:0]                      m09_axi_arlen  ,
   input  logic                              m09_axi_rvalid ,
   output logic                              m09_axi_rready ,
   input  logic [C_M09_AXI_DATA_WIDTH-1:0]   m09_axi_rdata  ,
   input  logic                              m09_axi_rlast  ,

   output logic                              m10_axi_awvalid,
   input  logic                              m10_axi_awready,
   output logic [C_M10_AXI_ADDR_WIDTH-1:0]   m10_axi_awaddr ,
   output logic [8-1:0]                      m10_axi_awlen  ,
   output logic                              m10_axi_wvalid ,
   input  logic                              m10_axi_wready ,
   output logic [C_M10_AXI_DATA_WIDTH-1:0]   m10_axi_wdata  ,
   output logic [C_M10_AXI_DATA_WIDTH/8-1:0] m10_axi_wstrb  ,
   output logic                              m10_axi_wlast  ,
   input  logic                              m10_axi_bvalid ,
   output logic                              m10_axi_bready ,
   output logic                              m10_axi_arvalid,
   input  logic                              m10_axi_arready,
   output logic [C_M10_AXI_ADDR_WIDTH-1:0]   m10_axi_araddr ,
   output logic [8-1:0]                      m10_axi_arlen  ,
   input  logic                              m10_axi_rvalid ,
   output logic                              m10_axi_rready ,
   input  logic [C_M10_AXI_DATA_WIDTH-1:0]   m10_axi_rdata  ,
   input  logic                              m10_axi_rlast  ,

   output logic                              m11_axi_awvalid,
   input  logic                              m11_axi_awready,
   output logic [C_M11_AXI_ADDR_WIDTH-1:0]   m11_axi_awaddr ,
   output logic [8-1:0]                      m11_axi_awlen  ,
   output logic                              m11_axi_wvalid ,
   input  logic                              m11_axi_wready ,
   output logic [C_M11_AXI_DATA_WIDTH-1:0]   m11_axi_wdata  ,
   output logic [C_M11_AXI_DATA_WIDTH/8-1:0] m11_axi_wstrb  ,
   output logic                              m11_axi_wlast  ,
   input  logic                              m11_axi_bvalid ,
   output logic                              m11_axi_bready ,
   output logic                              m11_axi_arvalid,
   input  logic                              m11_axi_arready,
   output logic [C_M11_AXI_ADDR_WIDTH-1:0]   m11_axi_araddr ,
   output logic [8-1:0]                      m11_axi_arlen  ,
   input  logic                              m11_axi_rvalid ,
   output logic                              m11_axi_rready ,
   input  logic [C_M11_AXI_DATA_WIDTH-1:0]   m11_axi_rdata  ,
   input  logic                              m11_axi_rlast  ,

   output logic                              m12_axi_awvalid,
   input  logic                              m12_axi_awready,
   output logic [C_M12_AXI_ADDR_WIDTH-1:0]   m12_axi_awaddr ,
   output logic [8-1:0]                      m12_axi_awlen  ,
   output logic                              m12_axi_wvalid ,
   input  logic                              m12_axi_wready ,
   output logic [C_M12_AXI_DATA_WIDTH-1:0]   m12_axi_wdata  ,
   output logic [C_M12_AXI_DATA_WIDTH/8-1:0] m12_axi_wstrb  ,
   output logic                              m12_axi_wlast  ,
   input  logic                              m12_axi_bvalid ,
   output logic                              m12_axi_bready ,
   output logic                              m12_axi_arvalid,
   input  logic                              m12_axi_arready,
   output logic [C_M12_AXI_ADDR_WIDTH-1:0]   m12_axi_araddr ,
   output logic [8-1:0]                      m12_axi_arlen  ,
   input  logic                              m12_axi_rvalid ,
   output logic                              m12_axi_rready ,
   input  logic [C_M12_AXI_DATA_WIDTH-1:0]   m12_axi_rdata  ,
   input  logic                              m12_axi_rlast  ,

   output logic                              m13_axi_awvalid,
   input  logic                              m13_axi_awready,
   output logic [C_M13_AXI_ADDR_WIDTH-1:0]   m13_axi_awaddr ,
   output logic [8-1:0]                      m13_axi_awlen  ,
   output logic                              m13_axi_wvalid ,
   input  logic                              m13_axi_wready ,
   output logic [C_M13_AXI_DATA_WIDTH-1:0]   m13_axi_wdata  ,
   output logic [C_M13_AXI_DATA_WIDTH/8-1:0] m13_axi_wstrb  ,
   output logic                              m13_axi_wlast  ,
   input  logic                              m13_axi_bvalid ,
   output logic                              m13_axi_bready ,
   output logic                              m13_axi_arvalid,
   input  logic                              m13_axi_arready,
   output logic [C_M13_AXI_ADDR_WIDTH-1:0]   m13_axi_araddr ,
   output logic [8-1:0]                      m13_axi_arlen  ,
   input  logic                              m13_axi_rvalid ,
   output logic                              m13_axi_rready ,
   input  logic [C_M13_AXI_DATA_WIDTH-1:0]   m13_axi_rdata  ,
   input  logic                              m13_axi_rlast  ,

   output logic                              m14_axi_awvalid,
   input  logic                              m14_axi_awready,
   output logic [C_M14_AXI_ADDR_WIDTH-1:0]   m14_axi_awaddr ,
   output logic [8-1:0]                      m14_axi_awlen  ,
   output logic                              m14_axi_wvalid ,
   input  logic                              m14_axi_wready ,
   output logic [C_M14_AXI_DATA_WIDTH-1:0]   m14_axi_wdata  ,
   output logic [C_M14_AXI_DATA_WIDTH/8-1:0] m14_axi_wstrb  ,
   output logic                              m14_axi_wlast  ,
   input  logic                              m14_axi_bvalid ,
   output logic                              m14_axi_bready ,
   output logic                              m14_axi_arvalid,
   input  logic                              m14_axi_arready,
   output logic [C_M14_AXI_ADDR_WIDTH-1:0]   m14_axi_araddr ,
   output logic [8-1:0]                      m14_axi_arlen  ,
   input  logic                              m14_axi_rvalid ,
   output logic                              m14_axi_rready ,
   input  logic [C_M14_AXI_DATA_WIDTH-1:0]   m14_axi_rdata  ,
   input  logic                              m14_axi_rlast  ,

   output logic                              m15_axi_awvalid,
   input  logic                              m15_axi_awready,
   output logic [C_M15_AXI_ADDR_WIDTH-1:0]   m15_axi_awaddr ,
   output logic [8-1:0]                      m15_axi_awlen  ,
   output logic                              m15_axi_wvalid ,
   input  logic                              m15_axi_wready ,
   output logic [C_M15_AXI_DATA_WIDTH-1:0]   m15_axi_wdata  ,
   output logic [C_M15_AXI_DATA_WIDTH/8-1:0] m15_axi_wstrb  ,
   output logic                              m15_axi_wlast  ,
   input  logic                              m15_axi_bvalid ,
   output logic                              m15_axi_bready ,
   output logic                              m15_axi_arvalid,
   input  logic                              m15_axi_arready,
   output logic [C_M15_AXI_ADDR_WIDTH-1:0]   m15_axi_araddr ,
   output logic [8-1:0]                      m15_axi_arlen  ,
   input  logic                              m15_axi_rvalid ,
   output logic                              m15_axi_rready ,
   input  logic [C_M15_AXI_DATA_WIDTH-1:0]   m15_axi_rdata  ,
   input  logic                              m15_axi_rlast  ,

   output logic                              m16_axi_awvalid,
   input  logic                              m16_axi_awready,
   output logic [C_M06_AXI_ADDR_WIDTH-1:0]   m16_axi_awaddr ,
   output logic [8-1:0]                      m16_axi_awlen  ,
   output logic                              m16_axi_wvalid ,
   input  logic                              m16_axi_wready ,
   output logic [C_M06_AXI_DATA_WIDTH-1:0]   m16_axi_wdata  ,
   output logic [C_M06_AXI_DATA_WIDTH/8-1:0] m16_axi_wstrb  ,
   output logic                              m16_axi_wlast  ,
   input  logic                              m16_axi_bvalid ,
   output logic                              m16_axi_bready ,
   output logic                              m16_axi_arvalid,
   input  logic                              m16_axi_arready,
   output logic [C_M06_AXI_ADDR_WIDTH-1:0]   m16_axi_araddr ,
   output logic [8-1:0]                      m16_axi_arlen  ,
   input  logic                              m16_axi_rvalid ,
   output logic                              m16_axi_rready ,
   input  logic [C_M06_AXI_DATA_WIDTH-1:0]   m16_axi_rdata  ,
   input  logic                              m16_axi_rlast  ,

   output logic                              m17_axi_awvalid,
   input  logic                              m17_axi_awready,
   output logic [C_M07_AXI_ADDR_WIDTH-1:0]   m17_axi_awaddr ,
   output logic [8-1:0]                      m17_axi_awlen  ,
   output logic                              m17_axi_wvalid ,
   input  logic                              m17_axi_wready ,
   output logic [C_M07_AXI_DATA_WIDTH-1:0]   m17_axi_wdata  ,
   output logic [C_M07_AXI_DATA_WIDTH/8-1:0] m17_axi_wstrb  ,
   output logic                              m17_axi_wlast  ,
   input  logic                              m17_axi_bvalid ,
   output logic                              m17_axi_bready ,
   output logic                              m17_axi_arvalid,
   input  logic                              m17_axi_arready,
   output logic [C_M07_AXI_ADDR_WIDTH-1:0]   m17_axi_araddr ,
   output logic [8-1:0]                      m17_axi_arlen  ,
   input  logic                              m17_axi_rvalid ,
   output logic                              m17_axi_rready ,
   input  logic [C_M07_AXI_DATA_WIDTH-1:0]   m17_axi_rdata  ,
   input  logic                              m17_axi_rlast  ,

   output logic                              m18_axi_awvalid,
   input  logic                              m18_axi_awready,
   output logic [C_M08_AXI_ADDR_WIDTH-1:0]   m18_axi_awaddr ,
   output logic [8-1:0]                      m18_axi_awlen  ,
   output logic                              m18_axi_wvalid ,
   input  logic                              m18_axi_wready ,
   output logic [C_M08_AXI_DATA_WIDTH-1:0]   m18_axi_wdata  ,
   output logic [C_M08_AXI_DATA_WIDTH/8-1:0] m18_axi_wstrb  ,
   output logic                              m18_axi_wlast  ,
   input  logic                              m18_axi_bvalid ,
   output logic                              m18_axi_bready ,
   output logic                              m18_axi_arvalid,
   input  logic                              m18_axi_arready,
   output logic [C_M08_AXI_ADDR_WIDTH-1:0]   m18_axi_araddr ,
   output logic [8-1:0]                      m18_axi_arlen  ,
   input  logic                              m18_axi_rvalid ,
   output logic                              m18_axi_rready ,
   input  logic [C_M08_AXI_DATA_WIDTH-1:0]   m18_axi_rdata  ,
   input  logic                              m18_axi_rlast  ,

   output logic                              m19_axi_awvalid,
   input  logic                              m19_axi_awready,
   output logic [C_M09_AXI_ADDR_WIDTH-1:0]   m19_axi_awaddr ,
   output logic [8-1:0]                      m19_axi_awlen  ,
   output logic                              m19_axi_wvalid ,
   input  logic                              m19_axi_wready ,
   output logic [C_M09_AXI_DATA_WIDTH-1:0]   m19_axi_wdata  ,
   output logic [C_M09_AXI_DATA_WIDTH/8-1:0] m19_axi_wstrb  ,
   output logic                              m19_axi_wlast  ,
   input  logic                              m19_axi_bvalid ,
   output logic                              m19_axi_bready ,
   output logic                              m19_axi_arvalid,
   input  logic                              m19_axi_arready,
   output logic [C_M09_AXI_ADDR_WIDTH-1:0]   m19_axi_araddr ,
   output logic [8-1:0]                      m19_axi_arlen  ,
   input  logic                              m19_axi_rvalid ,
   output logic                              m19_axi_rready ,
   input  logic [C_M09_AXI_DATA_WIDTH-1:0]   m19_axi_rdata  ,
   input  logic                              m19_axi_rlast  ,

   output logic                              m20_axi_awvalid,
   input  logic                              m20_axi_awready,
   output logic [C_M10_AXI_ADDR_WIDTH-1:0]   m20_axi_awaddr ,
   output logic [8-1:0]                      m20_axi_awlen  ,
   output logic                              m20_axi_wvalid ,
   input  logic                              m20_axi_wready ,
   output logic [C_M10_AXI_DATA_WIDTH-1:0]   m20_axi_wdata  ,
   output logic [C_M10_AXI_DATA_WIDTH/8-1:0] m20_axi_wstrb  ,
   output logic                              m20_axi_wlast  ,
   input  logic                              m20_axi_bvalid ,
   output logic                              m20_axi_bready ,
   output logic                              m20_axi_arvalid,
   input  logic                              m20_axi_arready,
   output logic [C_M10_AXI_ADDR_WIDTH-1:0]   m20_axi_araddr ,
   output logic [8-1:0]                      m20_axi_arlen  ,
   input  logic                              m20_axi_rvalid ,
   output logic                              m20_axi_rready ,
   input  logic [C_M10_AXI_DATA_WIDTH-1:0]   m20_axi_rdata  ,
   input  logic                              m20_axi_rlast  ,

   output logic                              m21_axi_awvalid,
   input  logic                              m21_axi_awready,
   output logic [C_M11_AXI_ADDR_WIDTH-1:0]   m21_axi_awaddr ,
   output logic [8-1:0]                      m21_axi_awlen  ,
   output logic                              m21_axi_wvalid ,
   input  logic                              m21_axi_wready ,
   output logic [C_M11_AXI_DATA_WIDTH-1:0]   m21_axi_wdata  ,
   output logic [C_M11_AXI_DATA_WIDTH/8-1:0] m21_axi_wstrb  ,
   output logic                              m21_axi_wlast  ,
   input  logic                              m21_axi_bvalid ,
   output logic                              m21_axi_bready ,
   output logic                              m21_axi_arvalid,
   input  logic                              m21_axi_arready,
   output logic [C_M11_AXI_ADDR_WIDTH-1:0]   m21_axi_araddr ,
   output logic [8-1:0]                      m21_axi_arlen  ,
   input  logic                              m21_axi_rvalid ,
   output logic                              m21_axi_rready ,
   input  logic [C_M11_AXI_DATA_WIDTH-1:0]   m21_axi_rdata  ,
   input  logic                              m21_axi_rlast  ,

   output logic                              m22_axi_awvalid,
   input  logic                              m22_axi_awready,
   output logic [C_M02_AXI_ADDR_WIDTH-1:0]   m22_axi_awaddr ,
   output logic [8-1:0]                      m22_axi_awlen  ,
   output logic                              m22_axi_wvalid ,
   input  logic                              m22_axi_wready ,
   output logic [C_M02_AXI_DATA_WIDTH-1:0]   m22_axi_wdata  ,
   output logic [C_M02_AXI_DATA_WIDTH/8-1:0] m22_axi_wstrb  ,
   output logic                              m22_axi_wlast  ,
   input  logic                              m22_axi_bvalid ,
   output logic                              m22_axi_bready ,
   output logic                              m22_axi_arvalid,
   input  logic                              m22_axi_arready,
   output logic [C_M02_AXI_ADDR_WIDTH-1:0]   m22_axi_araddr ,
   output logic [8-1:0]                      m22_axi_arlen  ,
   input  logic                              m22_axi_rvalid ,
   output logic                              m22_axi_rready ,
   input  logic [C_M02_AXI_DATA_WIDTH-1:0]   m22_axi_rdata  ,
   input  logic                              m22_axi_rlast  ,

   output logic                              m23_axi_awvalid,
   input  logic                              m23_axi_awready,
   output logic [C_M03_AXI_ADDR_WIDTH-1:0]   m23_axi_awaddr ,
   output logic [8-1:0]                      m23_axi_awlen  ,
   output logic                              m23_axi_wvalid ,
   input  logic                              m23_axi_wready ,
   output logic [C_M03_AXI_DATA_WIDTH-1:0]   m23_axi_wdata  ,
   output logic [C_M03_AXI_DATA_WIDTH/8-1:0] m23_axi_wstrb  ,
   output logic                              m23_axi_wlast  ,
   input  logic                              m23_axi_bvalid ,
   output logic                              m23_axi_bready ,
   output logic                              m23_axi_arvalid,
   input  logic                              m23_axi_arready,
   output logic [C_M03_AXI_ADDR_WIDTH-1:0]   m23_axi_araddr ,
   output logic [8-1:0]                      m23_axi_arlen  ,
   input  logic                              m23_axi_rvalid ,
   output logic                              m23_axi_rready ,
   input  logic [C_M03_AXI_DATA_WIDTH-1:0]   m23_axi_rdata  ,
   input  logic                              m23_axi_rlast  ,

   output logic                              m24_axi_awvalid,
   input  logic                              m24_axi_awready,
   output logic [C_M04_AXI_ADDR_WIDTH-1:0]   m24_axi_awaddr ,
   output logic [8-1:0]                      m24_axi_awlen  ,
   output logic                              m24_axi_wvalid ,
   input  logic                              m24_axi_wready ,
   output logic [C_M04_AXI_DATA_WIDTH-1:0]   m24_axi_wdata  ,
   output logic [C_M04_AXI_DATA_WIDTH/8-1:0] m24_axi_wstrb  ,
   output logic                              m24_axi_wlast  ,
   input  logic                              m24_axi_bvalid ,
   output logic                              m24_axi_bready ,
   output logic                              m24_axi_arvalid,
   input  logic                              m24_axi_arready,
   output logic [C_M04_AXI_ADDR_WIDTH-1:0]   m24_axi_araddr ,
   output logic [8-1:0]                      m24_axi_arlen  ,
   input  logic                              m24_axi_rvalid ,
   output logic                              m24_axi_rready ,
   input  logic [C_M04_AXI_DATA_WIDTH-1:0]   m24_axi_rdata  ,
   input  logic                              m24_axi_rlast  ,

   output logic                              m25_axi_awvalid,
   input  logic                              m25_axi_awready,
   output logic [C_M05_AXI_ADDR_WIDTH-1:0]   m25_axi_awaddr ,
   output logic [8-1:0]                      m25_axi_awlen  ,
   output logic                              m25_axi_wvalid ,
   input  logic                              m25_axi_wready ,
   output logic [C_M05_AXI_DATA_WIDTH-1:0]   m25_axi_wdata  ,
   output logic [C_M05_AXI_DATA_WIDTH/8-1:0] m25_axi_wstrb  ,
   output logic                              m25_axi_wlast  ,
   input  logic                              m25_axi_bvalid ,
   output logic                              m25_axi_bready ,
   output logic                              m25_axi_arvalid,
   input  logic                              m25_axi_arready,
   output logic [C_M05_AXI_ADDR_WIDTH-1:0]   m25_axi_araddr ,
   output logic [8-1:0]                      m25_axi_arlen  ,
   input  logic                              m25_axi_rvalid ,
   output logic                              m25_axi_rready ,
   input  logic [C_M05_AXI_DATA_WIDTH-1:0]   m25_axi_rdata  ,
   input  logic                              m25_axi_rlast  ,

   output logic                              m26_axi_awvalid,
   input  logic                              m26_axi_awready,
   output logic [C_M06_AXI_ADDR_WIDTH-1:0]   m26_axi_awaddr ,
   output logic [8-1:0]                      m26_axi_awlen  ,
   output logic                              m26_axi_wvalid ,
   input  logic                              m26_axi_wready ,
   output logic [C_M06_AXI_DATA_WIDTH-1:0]   m26_axi_wdata  ,
   output logic [C_M06_AXI_DATA_WIDTH/8-1:0] m26_axi_wstrb  ,
   output logic                              m26_axi_wlast  ,
   input  logic                              m26_axi_bvalid ,
   output logic                              m26_axi_bready ,
   output logic                              m26_axi_arvalid,
   input  logic                              m26_axi_arready,
   output logic [C_M06_AXI_ADDR_WIDTH-1:0]   m26_axi_araddr ,
   output logic [8-1:0]                      m26_axi_arlen  ,
   input  logic                              m26_axi_rvalid ,
   output logic                              m26_axi_rready ,
   input  logic [C_M06_AXI_DATA_WIDTH-1:0]   m26_axi_rdata  ,
   input  logic                              m26_axi_rlast  ,

   output logic                              m27_axi_awvalid,
   input  logic                              m27_axi_awready,
   output logic [C_M07_AXI_ADDR_WIDTH-1:0]   m27_axi_awaddr ,
   output logic [8-1:0]                      m27_axi_awlen  ,
   output logic                              m27_axi_wvalid ,
   input  logic                              m27_axi_wready ,
   output logic [C_M07_AXI_DATA_WIDTH-1:0]   m27_axi_wdata  ,
   output logic [C_M07_AXI_DATA_WIDTH/8-1:0] m27_axi_wstrb  ,
   output logic                              m27_axi_wlast  ,
   input  logic                              m27_axi_bvalid ,
   output logic                              m27_axi_bready ,
   output logic                              m27_axi_arvalid,
   input  logic                              m27_axi_arready,
   output logic [C_M07_AXI_ADDR_WIDTH-1:0]   m27_axi_araddr ,
   output logic [8-1:0]                      m27_axi_arlen  ,
   input  logic                              m27_axi_rvalid ,
   output logic                              m27_axi_rready ,
   input  logic [C_M07_AXI_DATA_WIDTH-1:0]   m27_axi_rdata  ,
   input  logic                              m27_axi_rlast  ,

   output logic                              m28_axi_awvalid,
   input  logic                              m28_axi_awready,
   output logic [C_M08_AXI_ADDR_WIDTH-1:0]   m28_axi_awaddr ,
   output logic [8-1:0]                      m28_axi_awlen  ,
   output logic                              m28_axi_wvalid ,
   input  logic                              m28_axi_wready ,
   output logic [C_M08_AXI_DATA_WIDTH-1:0]   m28_axi_wdata  ,
   output logic [C_M08_AXI_DATA_WIDTH/8-1:0] m28_axi_wstrb  ,
   output logic                              m28_axi_wlast  ,
   input  logic                              m28_axi_bvalid ,
   output logic                              m28_axi_bready ,
   output logic                              m28_axi_arvalid,
   input  logic                              m28_axi_arready,
   output logic [C_M08_AXI_ADDR_WIDTH-1:0]   m28_axi_araddr ,
   output logic [8-1:0]                      m28_axi_arlen  ,
   input  logic                              m28_axi_rvalid ,
   output logic                              m28_axi_rready ,
   input  logic [C_M08_AXI_DATA_WIDTH-1:0]   m28_axi_rdata  ,
   input  logic                              m28_axi_rlast  ,

   output logic                              m29_axi_awvalid,
   input  logic                              m29_axi_awready,
   output logic [C_M09_AXI_ADDR_WIDTH-1:0]   m29_axi_awaddr ,
   output logic [8-1:0]                      m29_axi_awlen  ,
   output logic                              m29_axi_wvalid ,
   input  logic                              m29_axi_wready ,
   output logic [C_M09_AXI_DATA_WIDTH-1:0]   m29_axi_wdata  ,
   output logic [C_M09_AXI_DATA_WIDTH/8-1:0] m29_axi_wstrb  ,
   output logic                              m29_axi_wlast  ,
   input  logic                              m29_axi_bvalid ,
   output logic                              m29_axi_bready ,
   output logic                              m29_axi_arvalid,
   input  logic                              m29_axi_arready,
   output logic [C_M09_AXI_ADDR_WIDTH-1:0]   m29_axi_araddr ,
   output logic [8-1:0]                      m29_axi_arlen  ,
   input  logic                              m29_axi_rvalid ,
   output logic                              m29_axi_rready ,
   input  logic [C_M09_AXI_DATA_WIDTH-1:0]   m29_axi_rdata  ,
   input  logic                              m29_axi_rlast  ,

   output logic                              m30_axi_awvalid,
   input  logic                              m30_axi_awready,
   output logic [C_M10_AXI_ADDR_WIDTH-1:0]   m30_axi_awaddr ,
   output logic [8-1:0]                      m30_axi_awlen  ,
   output logic                              m30_axi_wvalid ,
   input  logic                              m30_axi_wready ,
   output logic [C_M10_AXI_DATA_WIDTH-1:0]   m30_axi_wdata  ,
   output logic [C_M10_AXI_DATA_WIDTH/8-1:0] m30_axi_wstrb  ,
   output logic                              m30_axi_wlast  ,
   input  logic                              m30_axi_bvalid ,
   output logic                              m30_axi_bready ,
   output logic                              m30_axi_arvalid,
   input  logic                              m30_axi_arready,
   output logic [C_M10_AXI_ADDR_WIDTH-1:0]   m30_axi_araddr ,
   output logic [8-1:0]                      m30_axi_arlen  ,
   input  logic                              m30_axi_rvalid ,
   output logic                              m30_axi_rready ,
   input  logic [C_M10_AXI_DATA_WIDTH-1:0]   m30_axi_rdata  ,
   input  logic                              m30_axi_rlast  ,

   output logic                              m31_axi_awvalid,
   input  logic                              m31_axi_awready,
   output logic [C_M11_AXI_ADDR_WIDTH-1:0]   m31_axi_awaddr ,
   output logic [8-1:0]                      m31_axi_awlen  ,
   output logic                              m31_axi_wvalid ,
   input  logic                              m31_axi_wready ,
   output logic [C_M11_AXI_DATA_WIDTH-1:0]   m31_axi_wdata  ,
   output logic [C_M11_AXI_DATA_WIDTH/8-1:0] m31_axi_wstrb  ,
   output logic                              m31_axi_wlast  ,
   input  logic                              m31_axi_bvalid ,
   output logic                              m31_axi_bready ,
   output logic                              m31_axi_arvalid,
   input  logic                              m31_axi_arready,
   output logic [C_M11_AXI_ADDR_WIDTH-1:0]   m31_axi_araddr ,
   output logic [8-1:0]                      m31_axi_arlen  ,
   input  logic                              m31_axi_rvalid ,
   output logic                              m31_axi_rready ,
   input  logic [C_M11_AXI_DATA_WIDTH-1:0]   m31_axi_rdata  ,
   input  logic                              m31_axi_rlast  ,

   input  logic                                    s_axi_control_awvalid,
   output logic                                    s_axi_control_awready,
   input  logic [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_awaddr ,
   input  logic                                    s_axi_control_wvalid ,
   output logic                                    s_axi_control_wready ,
   input  logic [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_wdata  ,
   input  logic [C_S_AXI_CONTROL_DATA_WIDTH/8-1:0] s_axi_control_wstrb  ,
   input  logic                                    s_axi_control_arvalid,
   output logic                                    s_axi_control_arready,
   input  logic [C_S_AXI_CONTROL_ADDR_WIDTH-1:0]   s_axi_control_araddr ,
   output logic                                    s_axi_control_rvalid ,
   input  logic                                    s_axi_control_rready ,
   output logic [C_S_AXI_CONTROL_DATA_WIDTH-1:0]   s_axi_control_rdata  ,
   output logic [2-1:0]                            s_axi_control_rresp  ,
   output logic                                    s_axi_control_bvalid ,
   input  logic                                    s_axi_control_bready ,
   output logic [2-1:0]                            s_axi_control_bresp  ,

   output logic                                    interrupt            
   );

  import config_pkg::*;

  point_t [NLANE-1:0][1:0] x_dma_to_ntt;
  logic [NLANE-1:0]        valid_dma_to_ntt;

  point_t [NLANE-1:0][1:0] x_ntt_to_dma;
  logic [NLANE-1:0]        valid_ntt_to_dma;

  logic                    pass1;


  // ----------------------------------------------------------------------
  // Create ntt reset from dma reset (async assert, sync deassert)

   logic ntt_rst_n;
   cdc_sync cdc_sync_ntt_rst
     ( .clk_i(ntt_clk_i),
       .rst_ni(dma_rst_ni),
       .i(1'b1),
       .o(ntt_rst_n) );

   (* dont_touch = "true" *) logic ntt_rst_dt_qn;
   logic ntt_rst_qn;
   always_ff @(posedge ntt_clk_i) begin
      ntt_rst_dt_qn <= ntt_rst_n;
      ntt_rst_qn <= ntt_rst_dt_qn;
   end

  // ----------------------------------------------------------------------
  // AXI4-Lite subordinate interface for CSRs

  logic areset = 1'b0;
  logic ap_start;
  logic ap_idle;
  logic ap_start_pulse;
  logic ap_done;
  logic ap_ready;

  enum logic [1:0] {IDLE, BUSY, DONE} state_q, state_d;

  always @(posedge dma_clk_i) begin
    if (areset) state_q <= IDLE;
    else        state_q <= state_d;
  end

  always_comb begin
    state_d  = state_q;
    ap_ready = ap_done;

    case (state_q)
      IDLE: if (ap_start) state_d = BUSY;
      BUSY: if (ap_done)  state_d = DONE;
      DONE:               state_d = IDLE;
    endcase

    ap_idle        = (state_d == IDLE);
    ap_start_pulse = (state_q == IDLE) && ap_start;
  end
   
  // Register and invert reset signal.
  always @(posedge dma_clk_i) begin
    areset <= ~dma_rst_ni;
  end

  // Ready Logic (non-pipelined case).
  logic [1-1:0]                      chicken_bits   ;
  logic [64-1:0]                     axi00_ptr0     ;
  logic [64-1:0]                     axi01_ptr0     ;
  logic [64-1:0]                     axi02_ptr0     ;
  logic [64-1:0]                     axi03_ptr0     ;
  logic [64-1:0]                     axi04_ptr0     ;
  logic [64-1:0]                     axi05_ptr0     ;
  logic [64-1:0]                     axi06_ptr0     ;
  logic [64-1:0]                     axi07_ptr0     ;
  logic [64-1:0]                     axi08_ptr0     ;
  logic [64-1:0]                     axi09_ptr0     ;
  logic [64-1:0]                     axi10_ptr0     ;
  logic [64-1:0]                     axi11_ptr0     ;
  logic [64-1:0]                     axi12_ptr0     ;
  logic [64-1:0]                     axi13_ptr0     ;
  logic [64-1:0]                     axi14_ptr0     ;
  logic [64-1:0]                     axi15_ptr0     ;
  logic [64-1:0]                     axi16_ptr0     ;
  logic [64-1:0]                     axi17_ptr0     ;
  logic [64-1:0]                     axi18_ptr0     ;
  logic [64-1:0]                     axi19_ptr0     ;
  logic [64-1:0]                     axi20_ptr0     ;
  logic [64-1:0]                     axi21_ptr0     ;
  logic [64-1:0]                     axi22_ptr0     ;
  logic [64-1:0]                     axi23_ptr0     ;
  logic [64-1:0]                     axi24_ptr0     ;
  logic [64-1:0]                     axi25_ptr0     ;
  logic [64-1:0]                     axi26_ptr0     ;
  logic [64-1:0]                     axi27_ptr0     ;
  logic [64-1:0]                     axi28_ptr0     ;
  logic [64-1:0]                     axi29_ptr0     ;
  logic [64-1:0]                     axi30_ptr0     ;
  logic [64-1:0]                     axi31_ptr0     ;

  csr
    #(
      .C_S_AXI_ADDR_WIDTH ( C_S_AXI_CONTROL_ADDR_WIDTH ),
      .C_S_AXI_DATA_WIDTH ( C_S_AXI_CONTROL_DATA_WIDTH )
      ) _csr
      (
       .ACLK       ( dma_clk_i             ),
       .ARESET     ( areset                ),
       .ACLK_EN    ( 1'b1                  ),
       .AWVALID    ( s_axi_control_awvalid ),
       .AWREADY    ( s_axi_control_awready ),
       .AWADDR     ( s_axi_control_awaddr  ),
       .WVALID     ( s_axi_control_wvalid  ),
       .WREADY     ( s_axi_control_wready  ),
       .WDATA      ( s_axi_control_wdata   ),
       .WSTRB      ( s_axi_control_wstrb   ),
       .ARVALID    ( s_axi_control_arvalid ),
       .ARREADY    ( s_axi_control_arready ),
       .ARADDR     ( s_axi_control_araddr  ),
       .RVALID     ( s_axi_control_rvalid  ),
       .RREADY     ( s_axi_control_rready  ),
       .RDATA      ( s_axi_control_rdata   ),
       .RRESP      ( s_axi_control_rresp   ),
       .BVALID     ( s_axi_control_bvalid  ),
       .BREADY     ( s_axi_control_bready  ),
       .BRESP      ( s_axi_control_bresp   ),
       .ap_start   ( ap_start              ),
       .ap_done    ( ap_done               ),
       .ap_ready   ( ap_ready              ),
       .ap_idle    ( ap_idle               ),
       .*
       );

  // ----------------------------------------------------------------------
  // Point read and write DMA

   beat_id_t dbg_ch0_wbeat;
   beat_id_t dbg_ch0_rbeat;
   beat_id_t dbg_ch8_wbeat;
   beat_id_t dbg_ch8_rbeat;
   logic [15:0] dbg_wstep;
   logic [15:0] dbg_rstep;

  dma
    #(
      .C_M00_AXI_ADDR_WIDTH ( C_M00_AXI_ADDR_WIDTH ),
      .C_M00_AXI_DATA_WIDTH ( C_M00_AXI_DATA_WIDTH ),
      .C_M01_AXI_ADDR_WIDTH ( C_M01_AXI_ADDR_WIDTH ),
      .C_M01_AXI_DATA_WIDTH ( C_M01_AXI_DATA_WIDTH ),
      .C_M02_AXI_ADDR_WIDTH ( C_M02_AXI_ADDR_WIDTH ),
      .C_M02_AXI_DATA_WIDTH ( C_M02_AXI_DATA_WIDTH ),
      .C_M03_AXI_ADDR_WIDTH ( C_M03_AXI_ADDR_WIDTH ),
      .C_M03_AXI_DATA_WIDTH ( C_M03_AXI_DATA_WIDTH ),
      .C_M04_AXI_ADDR_WIDTH ( C_M04_AXI_ADDR_WIDTH ),
      .C_M04_AXI_DATA_WIDTH ( C_M04_AXI_DATA_WIDTH ),
      .C_M05_AXI_ADDR_WIDTH ( C_M05_AXI_ADDR_WIDTH ),
      .C_M05_AXI_DATA_WIDTH ( C_M05_AXI_DATA_WIDTH ),
      .C_M06_AXI_ADDR_WIDTH ( C_M06_AXI_ADDR_WIDTH ),
      .C_M06_AXI_DATA_WIDTH ( C_M06_AXI_DATA_WIDTH ),
      .C_M07_AXI_ADDR_WIDTH ( C_M07_AXI_ADDR_WIDTH ),
      .C_M07_AXI_DATA_WIDTH ( C_M07_AXI_DATA_WIDTH ),
      .C_M08_AXI_ADDR_WIDTH ( C_M08_AXI_ADDR_WIDTH ),
      .C_M08_AXI_DATA_WIDTH ( C_M08_AXI_DATA_WIDTH ),
      .C_M09_AXI_ADDR_WIDTH ( C_M09_AXI_ADDR_WIDTH ),
      .C_M09_AXI_DATA_WIDTH ( C_M09_AXI_DATA_WIDTH ),
      .C_M10_AXI_ADDR_WIDTH ( C_M10_AXI_ADDR_WIDTH ),
      .C_M10_AXI_DATA_WIDTH ( C_M10_AXI_DATA_WIDTH ),
      .C_M11_AXI_ADDR_WIDTH ( C_M11_AXI_ADDR_WIDTH ),
      .C_M11_AXI_DATA_WIDTH ( C_M11_AXI_DATA_WIDTH ),
      .C_M12_AXI_ADDR_WIDTH ( C_M12_AXI_ADDR_WIDTH ),
      .C_M12_AXI_DATA_WIDTH ( C_M12_AXI_DATA_WIDTH ),
      .C_M13_AXI_ADDR_WIDTH ( C_M13_AXI_ADDR_WIDTH ),
      .C_M13_AXI_DATA_WIDTH ( C_M13_AXI_DATA_WIDTH ),
      .C_M14_AXI_ADDR_WIDTH ( C_M14_AXI_ADDR_WIDTH ),
      .C_M14_AXI_DATA_WIDTH ( C_M14_AXI_DATA_WIDTH ),
      .C_M15_AXI_ADDR_WIDTH ( C_M15_AXI_ADDR_WIDTH ),
      .C_M15_AXI_DATA_WIDTH ( C_M15_AXI_DATA_WIDTH ),
      .C_M16_AXI_ADDR_WIDTH ( C_M16_AXI_ADDR_WIDTH ),
      .C_M16_AXI_DATA_WIDTH ( C_M16_AXI_DATA_WIDTH ),
      .C_M17_AXI_ADDR_WIDTH ( C_M17_AXI_ADDR_WIDTH ),
      .C_M17_AXI_DATA_WIDTH ( C_M17_AXI_DATA_WIDTH ),
      .C_M18_AXI_ADDR_WIDTH ( C_M18_AXI_ADDR_WIDTH ),
      .C_M18_AXI_DATA_WIDTH ( C_M18_AXI_DATA_WIDTH ),
      .C_M19_AXI_ADDR_WIDTH ( C_M19_AXI_ADDR_WIDTH ),
      .C_M19_AXI_DATA_WIDTH ( C_M19_AXI_DATA_WIDTH ),
      .C_M20_AXI_ADDR_WIDTH ( C_M20_AXI_ADDR_WIDTH ),
      .C_M20_AXI_DATA_WIDTH ( C_M20_AXI_DATA_WIDTH ),
      .C_M21_AXI_ADDR_WIDTH ( C_M21_AXI_ADDR_WIDTH ),
      .C_M21_AXI_DATA_WIDTH ( C_M21_AXI_DATA_WIDTH ),
      .C_M22_AXI_ADDR_WIDTH ( C_M22_AXI_ADDR_WIDTH ),
      .C_M22_AXI_DATA_WIDTH ( C_M22_AXI_DATA_WIDTH ),
      .C_M23_AXI_ADDR_WIDTH ( C_M23_AXI_ADDR_WIDTH ),
      .C_M23_AXI_DATA_WIDTH ( C_M23_AXI_DATA_WIDTH ),
      .C_M24_AXI_ADDR_WIDTH ( C_M24_AXI_ADDR_WIDTH ),
      .C_M24_AXI_DATA_WIDTH ( C_M24_AXI_DATA_WIDTH ),
      .C_M25_AXI_ADDR_WIDTH ( C_M25_AXI_ADDR_WIDTH ),
      .C_M25_AXI_DATA_WIDTH ( C_M25_AXI_DATA_WIDTH ),
      .C_M26_AXI_ADDR_WIDTH ( C_M26_AXI_ADDR_WIDTH ),
      .C_M26_AXI_DATA_WIDTH ( C_M26_AXI_DATA_WIDTH ),
      .C_M27_AXI_ADDR_WIDTH ( C_M27_AXI_ADDR_WIDTH ),
      .C_M27_AXI_DATA_WIDTH ( C_M27_AXI_DATA_WIDTH ),
      .C_M28_AXI_ADDR_WIDTH ( C_M28_AXI_ADDR_WIDTH ),
      .C_M28_AXI_DATA_WIDTH ( C_M28_AXI_DATA_WIDTH ),
      .C_M29_AXI_ADDR_WIDTH ( C_M29_AXI_ADDR_WIDTH ),
      .C_M29_AXI_DATA_WIDTH ( C_M29_AXI_DATA_WIDTH ),
      .C_M30_AXI_ADDR_WIDTH ( C_M30_AXI_ADDR_WIDTH ),
      .C_M30_AXI_DATA_WIDTH ( C_M30_AXI_DATA_WIDTH ),
      .C_M31_AXI_ADDR_WIDTH ( C_M31_AXI_ADDR_WIDTH ),
      .C_M31_AXI_DATA_WIDTH ( C_M31_AXI_DATA_WIDTH )
      ) _dma
      (
       .ntt_clk_i,
       .ntt_rst_ni(ntt_rst_qn),
       .dma_clk_i,
       .dma_rst_ni,
       .pass1_o(pass1),
       .x_o(x_dma_to_ntt),
       .valid_o(valid_dma_to_ntt),
       .x_i(x_ntt_to_dma),
       .valid_i(valid_ntt_to_dma),
       .start_i(ap_start_pulse),
       .done_o(ap_done),
       .dbg_ch0_wbeat,
       .dbg_ch0_rbeat,
       .dbg_ch8_wbeat,
       .dbg_ch8_rbeat,
       .dbg_wstep,
       .dbg_rstep,
       .*
       );

  if (LOOPBACK) begin : gen_loopback

    // ----------------------------------------------------------------------
    // Loopback points.

`ifndef LOOPBACK_VALID_TEST
    assign x_ntt_to_dma = x_dma_to_ntt;
    assign valid_ntt_to_dma = valid_dma_to_ntt;
`else
    point_t [NLANE-1:0][1:0] points[$];

    always @(posedge ntt_clk_i) begin
      if (!dma_rst_ni) begin
        valid_ntt_to_dma <= '0;
        x_ntt_to_dma <= '0;
      end else begin

        // Pop points off the queue.
        if (points.size() == 0) begin
          valid_ntt_to_dma <= '0;
          x_ntt_to_dma <= 'x;
        end else if ($urandom_range(0, 15) != 0) begin
          valid_ntt_to_dma <= {NLANE{1'b1}};
          x_ntt_to_dma <= points.pop_back();
        end else begin
          valid_ntt_to_dma <= '0;
          x_ntt_to_dma <= 'x;
        end

        // Push new points onto the queue.
        if (valid_dma_to_ntt[0] === 1) begin
          points.push_front(x_dma_to_ntt);
        end
      end
    end
`endif

  end : gen_loopback

  else begin : gen_ntt

    // ----------------------------------------------------------------------
    // NTT

    ntt_top
      #(
        .NLEVEL(NLEVEL),
        .NLEVEL0(NLEVEL0),
        .NLANE(NLANE),
        .NOP(NOP)
        ) _ntt
        (
         .rst_ni(ntt_rst_n),
         .clk_i(ntt_clk_i),
         .pass1_i(pass1),
         .x_i(x_dma_to_ntt),
         .valid_i(valid_dma_to_ntt),
         .x_o(x_ntt_to_dma),
         .valid_o(valid_ntt_to_dma)
         );

  end: gen_ntt

`ifdef NEVER
   ila_dma0 i_ila_dma0 
     (
      .clk(ntt_clk_i),
      .probe0(pass1),
      .probe1(valid_dma_to_ntt[0]),
      .probe2(valid_ntt_to_dma[0]),
      .probe3(dbg_ch0_wbeat),
      .probe4(dbg_ch0_rbeat),
      .probe5(dbg_ch8_wbeat),
      .probe6(dbg_ch8_rbeat),
      .probe7(dbg_wstep),
      .probe8(dbg_rstep)
      );
`endif
   
endmodule
