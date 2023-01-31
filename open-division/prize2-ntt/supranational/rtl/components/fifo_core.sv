// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// FIFO core data storage
//
// Coded to infer FPGA block RAMs automatically.

module fifo_core
  #(
    parameter type DATA_t     = logic,
    parameter int  DEPTH      = 2,
    parameter bit  RESET      = 0,
    parameter bit  SYNC_RESET = 0,
    parameter type ADDR_t     = logic [$clog2(DEPTH)-1:0]
    )
  (
   input  logic  clk_i,
   input  logic  rst_ni,
   input  logic  we_i,
   input  DATA_t wdata_i,
   input  ADDR_t waddr_i,
   input  logic  re_i,
   input  ADDR_t raddr_i,
   output DATA_t rdata_o
   );

  DATA_t mem_q [DEPTH];
  DATA_t rdata_q; // Special CDC flop naming cdc_*_q

  if (!RESET) begin : gen_no_reset

    always_ff @(posedge clk_i) begin
      if (we_i) begin
        mem_q[waddr_i] <= wdata_i;
      end
    end

    always_ff @(posedge clk_i) begin
      if (re_i) begin
        rdata_q <= mem_q[raddr_i];
      end
    end

  end : gen_no_reset
  else if (SYNC_RESET) begin : gen_sync_reset

    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        mem_q <= '{default: DATA_t'('0)};
      end else if (we_i) begin
        mem_q[waddr_i] <= wdata_i;
      end
    end

    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        rdata_q <= DATA_t'('0);
      end else if (re_i) begin
        rdata_q <= mem_q[raddr_i];
      end
    end

  end : gen_sync_reset
  else begin : gen_async_reset

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        mem_q <= '{default: DATA_t'('0)};
      end else if (we_i) begin
        mem_q[waddr_i] <= wdata_i;
      end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        rdata_q <= DATA_t'('0);
      end else if (re_i) begin
        rdata_q <= mem_q[raddr_i];
      end
    end

  end : gen_async_reset

  assign rdata_o = rdata_q;

endmodule
