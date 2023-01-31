// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module dma_counter
  #(
    parameter int MAX      = 2,
    parameter int START    = 0,
    parameter int STEP     = 1,
    parameter int FLAG     = 0,
    parameter type COUNT_t = logic [$clog2(MAX)-1:0]
    )
  (
   input  logic   clk_i,
   input  logic   rst_ni,
   input  logic   restart_i,
   input  logic   step_i,
   output logic   done_o,
   output COUNT_t count_o,
   output logic   flag_o,
   input  COUNT_t wrap_i
   );

  COUNT_t counter_q, counter_d;
  COUNT_t mask;
  logic   done_d;
  logic   last;

  localparam int CMP_MSP = FLAG == 0 ? 0 : $clog2(FLAG+1)-1;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      counter_q <= START;
      done_o    <= '0;
    end else if (restart_i || step_i) begin
      counter_q <= counter_d;
      done_o    <= done_d;
    end
  end

  always_comb begin
    mask = '1;
    mask <<= $clog2(STEP);

    last = counter_q == (wrap_i & mask);

    if (restart_i) begin
      counter_d = START;
      done_d    = '0;
    end else if (!step_i) begin
      counter_d = counter_q;
      done_d    = done_o;
    end else if (last) begin
      counter_d = START;
      done_d    = !done_o;
    end else begin
      counter_d = counter_q + STEP;
      done_d    = done_o;
    end

    count_o = counter_q;

    if (FLAG == 0) begin : gen_flag0
      flag_o = 1'b0;
    end else begin : gen_flag
      flag_o = counter_q[CMP_MSP:0] == FLAG;
    end
  end

endmodule

