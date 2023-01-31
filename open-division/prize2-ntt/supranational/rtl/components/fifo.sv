// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Clock domain crossing FIFO

module fifo
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
   input  DATA_t wdata_i,
   input  logic  wvalid_i,
   output logic  wready_o,
   output DATA_t rdata_o,
   output logic  rvalid_o,
   input  logic  rready_i
   );

  case (DEPTH)
    0: begin : gen_depth_0

      always_comb begin
        rdata_o  = wdata_i;
        rvalid_o = wvalid_i;
        wready_o = rready_i;
      end

    end
    1: begin : gen_depth_1

      DATA_t rdata_d;
      logic  rvalid_d;

      always_comb begin
        wready_o = !rvalid_o || rready_i;

        rdata_d  = rdata_o;
        rvalid_d = rvalid_o;

        if (wvalid_i && wready_o) begin
          rdata_d  = wdata_i;
          rvalid_d = 1'b1;
        end else if (rvalid_o && rready_i) begin
          rvalid_d = 1'b0;
        end
      end

      if (!RESET) begin : gen_no_reset
        always_ff @(posedge clk_i) begin
          rdata_o  <= rdata_d;
          rvalid_o <= rvalid_d;
        end
      end : gen_no_reset

      else if (SYNC_RESET) begin : gen_sync_reset
        always_ff @(posedge clk_i) begin
          if (!rst_ni) begin
            rdata_o  <= DATA_t'('0);
            rvalid_o <= 1'b0;
          end else begin
            rdata_o  <= rdata_d;
            rvalid_o <= rvalid_d;
          end
        end
      end : gen_sync_reset

      else begin : gen_async_reset
        always_ff @(posedge clk_i or negedge rst_ni) begin
          if (!rst_ni) begin
            rdata_o  <= DATA_t'('0);
            rvalid_o <= 1'b0;
          end else begin
            rdata_o  <= rdata_d;
            rvalid_o <= rvalid_d;
          end
        end
      end : gen_async_reset

    end
    default: begin : gen_depth_n

      ADDR_t waddr, raddr;
      logic  we,    re;
      DATA_t rdata;

      fifo_ctrl
        #(
          .DEPTH(DEPTH),
          .SYNC_RESET(SYNC_RESET)
          ) _ctrl
          (
           .clk_i,
           .rst_ni,
           .wvalid_i,
           .wready_o,
           .waddr_o(waddr),
           .we_o(we),
           .rvalid_o,
           .rready_i,
           .raddr_o(raddr),
           .re_o(re)
           );

      fifo_core
        #(
          .DATA_t(DATA_t),
          .DEPTH(DEPTH),
          .RESET(RESET),
          .SYNC_RESET(SYNC_RESET)
          ) _core
          (
           .clk_i,
           .rst_ni,
           .we_i(we),
           .waddr_i(waddr),
           .wdata_i,
           .re_i(re),
           .raddr_i(raddr),
           .rdata_o(rdata)
           );

      assign rdata_o = rvalid_o ? rdata : '0;

    end
  endcase

endmodule
