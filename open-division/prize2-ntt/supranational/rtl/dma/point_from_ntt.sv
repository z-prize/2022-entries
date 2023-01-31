// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module point_from_ntt
  import config_pkg::*;
  (
   input  logic clk_i,
   input  logic rst_ni,

   input  point_t  [NLANE-1:0][1:0]       x_i,
   input  logic [NLANE-1:0]               valid_i,

   output logic    [NLANE-1:0][NPPCH-1:0] we_o,
   output point_t  [NLANE-1:0][NPPCH-1:0] wdata_o,
   output fine_t   [NLANE-1:0][NPPCH-1:0] waddr_o,
   output coarse_t [NLANE-1:0]            wcoarse_o,
   input  coarse_t [NLANE-1:0]            rcoarse_i
   );

  // ----------------------------------------------------------------------
  // Pointer synchronization and gray/bin conversion.

  coarse_t [NLANE-1:0] rcoarse_gray, rcoarse_d, rcoarse_q;
  coarse_t             wcoarse_gray;
  coarse_t             wcoarse;

  cdc_sync #(.WIDTH($bits(coarse_t)), .RESET(0)) _rcoarse_gray[NLANE-1:0]
    (.clk_i, .rst_ni, .i(rcoarse_i), .o(rcoarse_gray));

  gray_to_bin #(.WIDTH($bits(coarse_t))) _rcoarse_d[NLANE-1:0]
    (.i(rcoarse_gray), .o(rcoarse_d));

  always_ff @(posedge clk_i) begin
    rcoarse_q <= rcoarse_d;
    wcoarse_o <= {NLANE{wcoarse_gray}};
  end

  bin_to_gray #(.WIDTH($bits(coarse_t))) _wcoarse_gray
    (.i(wcoarse), .o(wcoarse_gray));

  // ----------------------------------------------------------------------
  // This engine pushes points to the FIFOs in a synchronous manner
  // using coarse flow control to minimize control paths across a wide
  // datapath.

  localparam int CYCLE_TO_COARSE = ($clog2(FIFO_FINE_DEPTH) +
                                    $clog2(N_CYCLES/N_BEATS));

  // ----------------------------------------------------------------------
  // Stage 1: Flop incoming data.
  
  point_t  [NLANE-1:0][1:0]       stage1_x_q;
  logic                           stage1_valid_q;
  cycle_id_t                      stage1_cycle_q, stage1_cycle_d;
  pass_id_t                       stage1_pass_q,  stage1_pass_d;
  shift_t                         stage1_shift_q, stage1_shift_d;

  always_ff @(posedge clk_i) begin
    stage1_x_q <= x_i;
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      stage1_valid_q <= 1'b0;
      stage1_cycle_q <= START_CYCLE;
      stage1_pass_q  <= START_PASS;
      stage1_shift_q <= '0;
    end else begin
      stage1_valid_q <= valid_i[0];
      stage1_cycle_q <= stage1_cycle_d;
      stage1_pass_q  <= stage1_pass_d;
      stage1_shift_q <= stage1_shift_d;
    end
  end

  always_comb begin
    if (!stage1_valid_q) begin
      // No transfer so hold state.
      stage1_cycle_d = stage1_cycle_q;
      stage1_pass_d  = stage1_pass_q;
    end else if (stage1_cycle_q == LAST_CYCLE) begin
      // Wrap cycles.
      stage1_cycle_d = START_CYCLE;
      if (stage1_pass_q == LAST_PASS) begin
        // Wrap passes.
        stage1_pass_d  = '0;
      end else begin
        // Next pass.
        stage1_pass_d  = stage1_pass_q + 1'b1;
      end
    end else begin
      // Increment cycles.
      stage1_cycle_d = stage1_cycle_q + 1'b1;
      stage1_pass_d  = stage1_pass_q;
    end

    stage1_shift_d = get_shift_from_cycle(stage1_cycle_d,
                                          POINT_WRITE);
  end

  // ----------------------------------------------------------------------
  // Stage 2: xbar/circular shift to reorient x.

  cycle_id_t [SHIFT_PIPE_DEPTH:0][1:0]                   shift_cycle;
  logic      [SHIFT_PIPE_DEPTH:0][1:0]                   shift_pass;
  logic      [SHIFT_PIPE_DEPTH:0]                        shift_valid;
  shift_t    [SHIFT_PIPE_DEPTH:0]                        shift_amount;
  point_t    [SHIFT_PIPE_DEPTH:0][NLANE/2-1:0][1:0][1:0] shift_x;

  always_comb begin
    shift_cycle [0] = stage1_cycle_q;
    shift_pass  [0] = stage1_pass_q;
    shift_amount[0] = stage1_shift_q;
    shift_valid [0] = stage1_valid_q;
    shift_x     [0] = stage1_x_q;
  end

  point_t    [NLANE/2-1:0][1:0][1:0] shifted_q;
  cycle_id_t                         shifted_cycle_q;
  logic                              shifted_pass_q;
  logic                              shifted_valid_q;

  if (SHIFT_PIPE_DEPTH != 0) begin : _shift_pipe

    localparam MAX_SHIFT = (NLANE/2) / SHIFT_PIPE_DEPTH;

    for (genvar i = 0; i < SHIFT_PIPE_DEPTH; i++) begin : _shift

      always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
          shift_valid[i+1] <= 1'b0;
        end else begin
          shift_valid[i+1] <= shift_valid[i];
        end
      end

      shift_t shift_remaining;
      point_t [NLANE/2-1:0][1:0][1:0] shifted;
  
      always_comb begin
        shift_remaining = shift_amount[i];
        shifted         = shift_x     [i];

        for (int j = 0; j < MAX_SHIFT; j++) begin
          if (shift_remaining != 0) begin
            shift_remaining = shift_remaining - 1'b1;
            shifted = {shifted[0], shifted[NLANE/2-1:1]};
          end
        end
      end

      always_ff @(posedge clk_i) begin
        shift_cycle [i+1] <= shift_cycle[i];
        shift_pass  [i+1] <= shift_pass[i];
        shift_amount[i+1] <= shift_remaining;
        shift_x     [i+1] <= shifted;
      end    
    end

    always_comb begin
      shifted_cycle_q = shift_cycle[SHIFT_PIPE_DEPTH];
      shifted_pass_q  = shift_pass [SHIFT_PIPE_DEPTH];
      shifted_q       = shift_x    [SHIFT_PIPE_DEPTH];
      shifted_valid_q = shift_valid[SHIFT_PIPE_DEPTH];
    end

  end else begin : _shift

    point_t [NLANE/2-1:0][1:0][1:0] shifted;

    always_comb begin
      case (shift_amount[0])
        3'h0: shifted =  shift_x;
        3'h1: shifted = {shift_x[0][0], shift_x[0][NLANE/2-1:1]};
        3'h2: shifted = {shift_x[0][1], shift_x[0][NLANE/2-1:2]};
        3'h3: shifted = {shift_x[0][2], shift_x[0][NLANE/2-1:3]};
        3'h4: shifted = {shift_x[0][3], shift_x[0][NLANE/2-1:4]};
        3'h5: shifted = {shift_x[0][4], shift_x[0][NLANE/2-1:5]};
        3'h6: shifted = {shift_x[0][5], shift_x[0][NLANE/2-1:6]};
        3'h7: shifted = {shift_x[0][6], shift_x[0][NLANE/2-1:7]};
      endcase
    end

    always_ff @(posedge clk_i) begin
      shifted_cycle_q <= stage1_cycle_q;
      shifted_q       <= shifted;
    end

    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        shifted_valid_q <= 1'b0;
        shifted_pass_q  <= START_PASS;
      end else begin
        shifted_valid_q <= stage1_valid_q;
        shifted_pass_q  <= stage1_pass_q;
      end
    end
  end

  // ----------------------------------------------------------------------
  // Reorg and PPCH mux.

  point_t [NLANE/2-1:0][1:0][NPPCH-1:0] reorg;
  logic   [NLANE/2-1:0][1:0][NPPCH-1:0] reorg_valid;
  ppch_id_t                             ppch_sel;
  logic                                 swap;
  logic                                 lane_inner_swap;

  always_comb begin
    reorg       = '0;
    reorg_valid = '0;

    swap = get_swap_from_cycle(shifted_cycle_q);

    for (int lane_outer = 0; lane_outer < NLANE/2; lane_outer++) begin
      // Hardcoded "2" below is the square root of NPPCH.
      for (int ppch_idx = 0; ppch_idx < 2; ppch_idx++) begin
        ppch_sel = get_ppch_from_cycle(shifted_cycle_q,
                                       ppch_idx,
                                       shifted_pass_q);
        for (int lane_inner = 0; lane_inner < 2; lane_inner++) begin
          lane_inner_swap = logic'(lane_inner) ^ swap;

          // These should reduce to 3:1 muxes.
          // Also handle reorg in same logic.
          reorg_valid[lane_outer][lane_inner_swap][ppch_sel] = 1'b1;
          reorg      [lane_outer][lane_inner_swap][ppch_sel] =
              shifted_q[lane_outer][ppch_idx][lane_inner];
        end
      end
    end
  end

  // ----------------------------------------------------------------------
  // FIFO write interface.

  point_t  [NLANE-1:0][NPPCH-1:0] wdata_d;
  logic    [NLANE-1:0][NPPCH-1:0] we_d;
  fine_t   [NLANE-1:0][NPPCH-1:0] waddr_d;

  always_comb begin
    wcoarse   = coarse_t'(shifted_cycle_q >> CYCLE_TO_COARSE);
    wdata_d   = reorg;
    we_d      = shifted_valid_q ? reorg_valid : '0;

    for (int lane = 0; lane < NLANE; lane++) begin
      waddr_d[lane] = {NPPCH{get_fine_from_cycle(shifted_cycle_q,
                                                 lane,
                                                 shifted_pass_q)}};
    end
  end

  always_ff @(posedge clk_i) begin
    we_o      <= we_d;
    waddr_o   <= waddr_d;
    wdata_o   <= wdata_d;
  end

  // ----------------------------------------------------------------------

`ifndef SYNTHESIS
  logic    [NLANE-1:0] backpressure;
  coarse_t [NLANE-1:0] volume;

  always_comb begin
    for (int lane = 0; lane < NLANE; lane++) begin
      volume[lane] = coarse_t'(wcoarse - rcoarse_q[lane]);
      backpressure[lane] = volume[lane] == FIFO_COARSE_DEPTH;
    end
  end

  ASSERT_no_backpressure:
    assert property (@(posedge clk_i) disable iff (!rst_ni)
                     we_o !== '0 |-> (we_o & backpressure) === '0
                     ) else $fatal(1, "NTT-->DMA point backpressure.");
`endif

endmodule
