// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module point_to_ntt
  import config_pkg::*;
  (
   input  logic                           clk_i,
   input  logic                           rst_ni,
   output logic                           pass_o,

   output point_t  [NLANE-1:0][1:0]       x_o,
   (* dont_touch = "true" *) output logic    [NLANE-1:0]            valid_o,

   output logic    [NLANE-1:0][NPPCH-1:0] re_o,
   input  point_t  [NLANE-1:0][NPPCH-1:0] rdata_i,
   output fine_t   [NLANE-1:0][NPPCH-1:0] raddr_o,
   input  coarse_t [NLANE-1:0]            wcoarse_i,
   output coarse_t [NLANE-1:0]            rcoarse_o
   );

  logic                           re_d;
  fine_t   [NLANE-1:0][NPPCH-1:0] raddr_d;
  logic                           stage1_valid;
  cycle_id_t                      stage1_cycle_q, stage1_cycle_d;
  pass_id_t                       stage1_pass_q,  stage1_pass_d;

  point_t [NLANE/2-1:0][1:0][NPPCH-1:0] stage1_data_q;

  // ----------------------------------------------------------------------
  // Pointer synchronization and gray/bin conversion.

  coarse_t [NLANE-1:0] wcoarse_gray, wcoarse_d, wcoarse_q;
  coarse_t             rcoarse_gray;
  coarse_t             rcoarse;

  cdc_sync #(.WIDTH($bits(coarse_t)), .RESET(0)) _wcoarse_gray[NLANE-1:0]
    (.clk_i, .rst_ni, .i(wcoarse_i), .o(wcoarse_gray));

  gray_to_bin #(.WIDTH($bits(coarse_t))) _wcoarse_d[NLANE-1:0]
    (.i(wcoarse_gray), .o(wcoarse_d));

  always_ff @(posedge clk_i) begin
    wcoarse_q <= wcoarse_d;
    rcoarse_o <= {NLANE{rcoarse_gray}};
  end

  bin_to_gray #(.WIDTH($bits(coarse_t))) _rcoarse_gray
    (.i(rcoarse), .o(rcoarse_gray));

  // ----------------------------------------------------------------------
  // This engine pulls points from the FIFOs in a synchronous manner
  // using coarse flow control to minimize control paths across a wide
  // datapath.

  localparam int CYCLE_TO_COARSE = ($clog2(FIFO_FINE_DEPTH) +
                                    $clog2(N_CYCLES/N_BEATS));

  // ----------------------------------------------------------------------
  // Flop incoming data and generate control signals.

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      stage1_cycle_q <= START_CYCLE;
      stage1_pass_q  <= START_PASS;
    end else begin
      stage1_cycle_q <= stage1_cycle_d;
      stage1_pass_q  <= stage1_pass_d;
    end
  end

  always_ff @(posedge clk_i) begin
    re_o           <= {NLANE*NPPCH{re_d}};
    raddr_o        <= raddr_d;
  end

  always_comb begin
    rcoarse = coarse_t'(stage1_cycle_q >> CYCLE_TO_COARSE);

    // Flow control from FIFO read to NTT.
    stage1_valid = 1'b1;
    for (int lane = 0; lane < NLANE; lane++) begin
      stage1_valid &= wcoarse_q[lane] != rcoarse;
    end

    // FIFO read interface.
    re_d      = stage1_valid;

    for (int lane = 0; lane < NLANE; lane++) begin
      raddr_d[lane] = {NPPCH{get_fine_from_cycle(stage1_cycle_q,
                                                 lane,
                                                 stage1_pass_q)}};
    end

    if (!stage1_valid) begin
      // No transfer so hold state.
      stage1_cycle_d = stage1_cycle_q;
      stage1_pass_d  = stage1_pass_q;
    end else if (stage1_cycle_q == LAST_CYCLE) begin
      // Wrap cycles.
      stage1_cycle_d = START_CYCLE;
      if (stage1_pass_q == LAST_PASS) begin
        // Wrap passes.
        stage1_pass_d = '0;
      end else begin
        // Next pass.
        stage1_pass_d = stage1_pass_q + 1'b1;
      end
    end else begin
      // Increment cycles.
      stage1_cycle_d = stage1_cycle_q + 1'b1;
      stage1_pass_d  = stage1_pass_q;
    end
  end

  // ----------------------------------------------------------------------
  // Pipe along some control signals until FIFO read data is ready.

  logic                                 rden_valid_q;
  cycle_id_t                            rden_cycle_q;
  logic                                 rden_pass_q;

  logic                                 rdval_valid_q;
  cycle_id_t                            rdval_cycle_q;
  logic                                 rdval_pass_q;

  logic                                 rdcap_valid_q;
  cycle_id_t                            rdcap_cycle_q;
  logic                                 rdcap_pass_q;
  point_t [NLANE/2-1:0][1:0][NPPCH-1:0] rdcap_q;

  // FIFO read enable stage.
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rden_valid_q <= '0;
      rden_cycle_q <= START_CYCLE;
      rden_pass_q  <= START_PASS;
    end else begin
      rden_valid_q <= stage1_valid;
      rden_cycle_q <= stage1_cycle_q;
      rden_pass_q  <= stage1_pass_q;
    end
  end

  // FIFO read data valid stage.
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rdval_valid_q <= '0;
      rdval_cycle_q <= START_CYCLE;
      rdval_pass_q  <= START_PASS;
    end else begin
      rdval_valid_q <= rden_valid_q;
      rdval_cycle_q <= rden_cycle_q;
      rdval_pass_q  <= rden_pass_q;
    end
  end

  // FIFO read data captured stage.
  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      rdcap_valid_q <= '0;
      rdcap_cycle_q <= START_CYCLE;
      rdcap_pass_q  <= START_PASS;
    end else begin
      rdcap_valid_q <= rdval_valid_q;
      rdcap_cycle_q <= rdval_cycle_q;
      rdcap_pass_q  <= rdval_pass_q;
    end
  end

  always_ff @(posedge clk_i) begin
    rdcap_q <= rdata_i;
  end

  // ----------------------------------------------------------------------
  // Reorg and PPCH mux.

  point_t [NLANE/2-1:0][1:0][1:0] reorg_q, reorg_d;
  logic                           reorg_valid_q;
  logic                           reorg_pass_q;
  shift_t                         reorg_shift_q;
  ppch_id_t                       ppch_sel;
  logic                           swap;
  logic                           lane_inner_swap;

  always_comb begin
    swap = get_swap_from_cycle(rdcap_cycle_q);

    for (int lane_outer = 0; lane_outer < NLANE/2; lane_outer++) begin
      // Hardcoded "2" below is the square root of NPPCH.
      for (int ppch_idx = 0; ppch_idx < 2; ppch_idx++) begin
        ppch_sel = get_ppch_from_cycle(rdcap_cycle_q,
                                       ppch_idx,
                                       rdcap_pass_q);

        for (int lane_inner = 0; lane_inner < 2; lane_inner++) begin
          lane_inner_swap = logic'(lane_inner) ^ swap;

          // These should reduce to 3:1 muxes.
          // Also handle reorg in same logic.
          reorg_d[lane_outer][ppch_idx][lane_inner_swap] =
              rdcap_q[lane_outer][lane_inner][ppch_sel];
        end
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      reorg_valid_q <= '0;
      reorg_pass_q  <= START_PASS;
    end else begin
      reorg_valid_q <= rdcap_valid_q;
      reorg_pass_q  <= rdcap_pass_q;
    end
  end

  always_ff @(posedge clk_i) begin
    reorg_q       = reorg_d;
    reorg_shift_q = get_shift_from_cycle(rdcap_cycle_q, POINT_READ);
  end

  // ----------------------------------------------------------------------
  // Circular shift to finalize x.

  logic   [SHIFT_PIPE_DEPTH:0]                        shift_pass;
  logic   [SHIFT_PIPE_DEPTH:0]                        shift_valid;
  shift_t [SHIFT_PIPE_DEPTH:0]                        shift_amount;
  point_t [SHIFT_PIPE_DEPTH:0][NLANE/2-1:0][1:0][1:0] shift_x;

  always_comb begin
    shift_amount[0] = reorg_shift_q;
    shift_pass  [0] = reorg_pass_q;
    shift_valid [0] = reorg_valid_q;
    shift_x     [0] = reorg_q;
  end

  if (SHIFT_PIPE_DEPTH != 0) begin : _shift_pipe

    localparam MAX_SHIFT = (NLANE/2) / SHIFT_PIPE_DEPTH;

    for (genvar i = 0; i < SHIFT_PIPE_DEPTH; i++) begin : _shift

      always_ff @(posedge clk_i) begin
        if (!rst_ni) begin
          shift_valid[i+1] <= 1'b0;
          shift_pass [i+1] <= START_PASS;
        end else begin
          shift_valid[i+1] <= shift_valid[i];
          shift_pass [i+1] <= shift_pass [i];
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
        shift_amount[i+1] <= shift_remaining;
        shift_x     [i+1] <= shifted;
      end    
    end

    always_comb begin
      x_o     = shift_x    [SHIFT_PIPE_DEPTH];
      pass_o  = shift_pass [SHIFT_PIPE_DEPTH];
    end

    always_ff @(posedge clk_i) begin
      valid_o <= {NLANE{shift_valid[SHIFT_PIPE_DEPTH-1]}};
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
      x_o <= shifted;
    end

    always_ff @(posedge clk_i) begin
      if (!rst_ni) begin
        valid_o <= {NLANE{1'b0}};
        pass_o  <= START_PASS;
      end else begin
        valid_o <= {NLANE{reorg_valid_q}};
        pass_o  <= reorg_pass_q;
      end
    end
  end

endmodule
