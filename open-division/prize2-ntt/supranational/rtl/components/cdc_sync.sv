// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Copyright Supranational LLC
//
// Universal synchronizer
//
// Note: Flops adhering to the cdc_*_q naming convention
// signal the input is from a different clock domain than
// the flop's clock.

module cdc_sync
  #(
    parameter int WIDTH = 1,
    parameter bit RESET = 1,
    parameter int RANK  = 2,
    parameter bit RAND  = 1,
    parameter bit [WIDTH-1:0] RESET_VALUE = '0
    )
  (
   input  logic             clk_i,
   input  logic             rst_ni,//NTH = 1'b1, // Default input might not work in some tools
   input  logic [WIDTH-1:0] i,
   output logic [WIDTH-1:0] o
   );

`ifndef SYNTHESIS
  assert #0 ((RANK >= 2) && (RANK <= 4)) else
    $fatal(1, "Rank must be between 2 and 4");
`endif

  // Give these flops a special instance name
  // using the "CDC" prefix so we can easily
  // script CDC reports in various EDA tools.
  logic [WIDTH-1:0] cdc_flop_q, cdc_flop_d;

  `ifdef SYNTHESIS

  assign cdc_flop_d = i;

  `else

  // ----------------------------------------------------------------------
  // Model synchronizer randomness for simulation

  if (RAND) begin : gen_rand_true

    // Synchronizer randomness enabled.
    logic [WIDTH-1:0] mask, ambiguous_bits, i_prev;

    always @(posedge clk_i or rst_ni or i) begin
      // Identify the ambiguous bits.
      mask = (i ^ i_prev);

      // Remember value for the next time.
      i_prev = i;

      // Randomize the ambiguous bits.
      ambiguous_bits = $urandom_range((2**WIDTH)-1, 0);

      // Combine the known bits and ambiguous bits.
      cdc_flop_d <= (i & ~mask) | (ambiguous_bits & mask);
    end

  end else begin : gen_rand_false

    // Synchronizer randomness disabled.
    assign cdc_flop_d = i;

  end

  `endif

  // Synchronizer flops
  logic [RANK-1:0][WIDTH-1:0] sync_q;
  assign sync_q[0] = cdc_flop_q;

  if (RESET) begin : gen_reset_true

    // Reset required for flop, async-reset assumed.
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if (!rst_ni) begin
	cdc_flop_q       <= RESET_VALUE;
        sync_q[RANK-1:1] <= '{default: RESET_VALUE};
      end
      else begin
        cdc_flop_q       <= cdc_flop_d;
        sync_q[RANK-1:1] <= sync_q[RANK-2:0];
      end
    end

  end else begin : gen_reset_false

    // No reset required for flops.
    always_ff @(posedge clk_i) begin
      cdc_flop_q <= cdc_flop_d;
      sync_q[RANK-1:1] <= sync_q[RANK-2:0];
    end

  end

  assign o = sync_q[RANK-1];

endmodule
