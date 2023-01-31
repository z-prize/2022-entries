// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// FIFO control logic

module fifo_ctrl
  #(
    parameter int  DEPTH      = 2,
    parameter int  ADJ_LSB    = 0,
    parameter bit  SYNC_RESET = 0,
    parameter type ADDR_t     = logic [$clog2(DEPTH)-1:0],
    parameter type PTR_t      = logic [$clog2(DEPTH):0]
    )
  (
   input  logic  clk_i,
   input  logic  rst_ni,
   input  logic  wvalid_i,
   output logic  wready_o,
   output ADDR_t waddr_o,
   output logic  we_o,
   output logic  rvalid_o,
   input  logic  rready_i,
   output ADDR_t raddr_o,
   output logic  re_o
   );

`ifndef SYNTHESIS
  assert #0 (DEPTH >=2)
    else $fatal(1, "FIFO must have a depth of 2 or more");
`endif

  ADDR_t waddr_q,     waddr_d;
  ADDR_t waddr_wrap;
  PTR_t  wptr_q,      wptr_d;
  PTR_t  wptr_plus1;

  logic  rempty;
  logic  rvalid_q,    rvalid_d;
  ADDR_t raddr_q,     raddr_d;
  ADDR_t raddr_wrap;
  PTR_t  rptr_q,      rptr_d;
  PTR_t  rptr_plus1;

  PTR_t  wptr_minus_rptr;

  localparam int DEPTH_ADJ  = ADJ_LSB == 0 ? 0 : 2**ADJ_LSB;
  localparam int SAFE_DEPTH = DEPTH - DEPTH_ADJ;

  // ----------------------------------------------------------------------
  // Write side

  assign wptr_plus1 = wptr_q + 1'b1;

  always_comb begin
    wptr_minus_rptr = wptr_q - rptr_q;

    // Wrap or increment to advance.
    if (waddr_q == (DEPTH-1)) begin
      waddr_wrap = '0;
    end else begin
      waddr_wrap = waddr_q + 1'b1;
    end

    // Not ready for writes if full.
    wready_o = wptr_minus_rptr != PTR_t'(SAFE_DEPTH);

    we_o = wvalid_i && wready_o;
    waddr_o = waddr_q;

    // Push the FIFO.
    if (wvalid_i && wready_o) begin
      wptr_d      = wptr_plus1;
      waddr_d     = waddr_wrap;
    end else begin
      wptr_d      = wptr_q;
      waddr_d     = waddr_q;
    end
  end

  // ----------------------------------------------------------------------
  // Read side

  always_comb begin
    rvalid_o   = rvalid_q;
    rempty     = rptr_q[$clog2(DEPTH):ADJ_LSB] ==
                 wptr_q[$clog2(DEPTH):ADJ_LSB];
    rptr_plus1 = rptr_q + 1'b1;

    // Wrap or increment to advance.
    if (raddr_q == (DEPTH-1)) begin
      raddr_wrap = '0;
    end else begin
      raddr_wrap = raddr_q + 1'b1;
    end

    raddr_o = raddr_wrap;

    if (!rvalid_q && !rempty) begin
      // Prime the read when exiting empty state.
      re_o     = 1'b1;
      raddr_d  = raddr_wrap;
      rptr_d   = rptr_plus1;
      rvalid_d = 1'b1;
    end else if (rvalid_q && rready_i && rempty) begin
      // Don't read past empty.
      re_o     = 1'b0;
      raddr_d  = raddr_q;
      rptr_d   = rptr_q;
      rvalid_d = 1'b0;
    end else if (rvalid_q && rready_i) begin
      // Normal FIFO read advance.
      re_o     = 1'b1;
      raddr_d  = raddr_wrap;
      rptr_d   = rptr_plus1;
      rvalid_d = rvalid_q;      
    end else begin
      // Hold state.
      re_o     = 1'b0;
      raddr_d  = raddr_q;
      rptr_d   = rptr_q;
      rvalid_d = rvalid_q;
    end
  end

  // ----------------------------------------------------------------------
  // Flops.

  if (SYNC_RESET) begin : gen_sync_reset
    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        rptr_q      <= '0;
        raddr_q     <= DEPTH-1;
        rvalid_q    <= '0;
        wptr_q      <= '0;
        waddr_q     <= '0;
      end else begin
        rptr_q      <= rptr_d;
        raddr_q     <= raddr_d;
        rvalid_q    <= rvalid_d;
        wptr_q      <= wptr_d;
        waddr_q     <= waddr_d;
      end
    end
  end : gen_sync_reset

  else begin : gen_async_reset
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
        rptr_q      <= '0;
        raddr_q     <= DEPTH-1;
        rvalid_q    <= '0;
        wptr_q      <= '0;
        waddr_q     <= '0;
      end else begin
        rptr_q      <= rptr_d;
        raddr_q     <= raddr_d;
        rvalid_q    <= rvalid_d;
        wptr_q      <= wptr_d;
        waddr_q     <= waddr_d;
      end
    end
  end : gen_async_reset

  // ----------------------------------------------------------------------
  // Assertions.

`ifndef SYNTHESIS
  ASSERT_no_underflow:
    assert property (@(posedge clk_i) disable iff (!rst_ni)
                     re_o === 1 |-> rempty === 0
                     ) else $error("FIFO underflow.");

  ASSERT_no_overflow:
    assert property (@(posedge clk_i) disable iff (!rst_ni)
                     we_o === 1 |-> wready_o === 1
                     ) else $fatal(1, "FIFO overflow.");
`endif

endmodule
