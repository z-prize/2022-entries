// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module point_dma_r_channel
  import config_pkg::*;
  #(
    parameter int ID = 0,
    parameter int C_M_AXI_ADDR_WIDTH = 64,
    parameter int C_M_AXI_DATA_WIDTH = 256
    )
  (
   input  logic clk_i,
   input  logic rst_ni,
   input  logic start_i,
   output logic done_o,
   output logic hold_o,
   input  logic release_i,

   output point_t [NPPCH-1:0]             wdata_o,
   output logic   [NPPCH-1:0]             we_o,
   output fine_t  [NPPCH-1:0]             waddr_o,
   output coarse_t                        wcoarse_o,
   input  coarse_t                        rcoarse_i,

   output beat_id_t                       dbg_rbeat,
   output logic                           dbg_rstep,

   // Tool generated ports.
   output logic                           m0_axi_arvalid,
   input  logic                           m0_axi_arready,
   output logic [C_M_AXI_ADDR_WIDTH-1:0]  m0_axi_araddr,
   output logic [8-1:0]                   m0_axi_arlen,
   input  logic                           m0_axi_rvalid,
   output logic                           m0_axi_rready,
   input  logic [C_M_AXI_DATA_WIDTH-1:0]  m0_axi_rdata,
   input  logic                           m0_axi_rlast,

   output logic                           m1_axi_arvalid,
   input  logic                           m1_axi_arready,
   output logic [C_M_AXI_ADDR_WIDTH-1:0]  m1_axi_araddr,
   output logic [8-1:0]                   m1_axi_arlen,
   input  logic                           m1_axi_rvalid,
   output logic                           m1_axi_rready,
   input  logic [C_M_AXI_DATA_WIDTH-1:0]  m1_axi_rdata,
   input  logic                           m1_axi_rlast,

   input  logic [C_M_AXI_ADDR_WIDTH-1:0]  ctrl_addr_offset0,
   input  logic [C_M_AXI_ADDR_WIDTH-1:0]  ctrl_addr_offset1
   );

  import axi_hbm_pkg::*;

  localparam int LEN  = DMA_LEN;
  localparam int LANE = ID / (N_HBM_PC / NLANE);

  typedef enum logic [1:0] {IDLE, BUSY, PASS, DONE} state_e;

  state_e   state_q,  state_d;
  pass_id_t pass_q,   pass_d;
  beat_id_t arbeat_q;
  beat_id_t rbeat_q;
  beat_id_t wrap;

  logic restart;
  logic utilized;

  coarse_t volume;
  logic    pause;
  addr_t   addr;

  // ----------------------------------------------------------------------
  // Pointer synchronization and gray/bin conversion.

  coarse_t rcoarse_gray, rcoarse_d, rcoarse_q;
  coarse_t wcoarse_gray;
  coarse_t wcoarse;

  cdc_sync #(.WIDTH($bits(coarse_t)), .RESET(0)) _rcoarse_gray
    (.clk_i, .rst_ni, .i(rcoarse_i), .o(rcoarse_gray));

  gray_to_bin #(.WIDTH($bits(coarse_t))) _rcoarse_d
    (.i(rcoarse_gray), .o(rcoarse_d));

  always_ff @(posedge clk_i) begin
    rcoarse_q <= rcoarse_d;
    wcoarse_o <= wcoarse_gray;
  end

  bin_to_gray #(.WIDTH($bits(coarse_t))) _wcoarse_gray
    (.i(wcoarse), .o(wcoarse_gray));

  // ----------------------------------------------------------------------
  // AXI burst tracking.

  logic ar_step, ar_done;
  dma_counter #(.MAX(N_BEATS), .STEP(LEN), .START(START_BEAT)) _counter_ar
    (
     .clk_i,
     .rst_ni,
     .restart_i(restart),
     .step_i(ar_step),
     .done_o(ar_done),
     .count_o(arbeat_q),
     .flag_o(),
     .wrap_i(wrap)
     );

  logic r_step, r_done;
  dma_counter #(.MAX(N_BEATS), .STEP(1), .START(START_BEAT)) _counter_r
    (
     .clk_i,
     .rst_ni,
     .restart_i(restart),
     .step_i(r_step),
     .done_o(r_done),
     .count_o(rbeat_q),
     .flag_o(),
     .wrap_i(wrap)
     );

  // ----------------------------------------------------------------------
  // AXI signal assignments.

  always_comb begin
    addr = get_addr_from_beat(arbeat_q, LANE, pass_q, POINT_READ);

    m0_axi_arvalid = (pass_q == 1'b0) && (state_q == BUSY) && !ar_done;
    m1_axi_arvalid = (pass_q == 1'b1) && (state_q == BUSY) && !ar_done;

    m0_axi_rready  = !pause;
    m1_axi_rready  = !pause;

    m0_axi_arlen   = LEN-1;
    m1_axi_arlen   = LEN-1;

    m0_axi_araddr = addr | $bits(m0_axi_araddr)'(ctrl_addr_offset0);
    m1_axi_araddr = addr | $bits(m1_axi_araddr)'(ctrl_addr_offset1);

    // Removed do to suspected HBM crossbar IP bug.
    // Trade addresses with the write channel paired with this one.
    // This uses the built in crossbar in the HBM IP to ping pong
    // between read and write buffers for first and second passes
    // thus saving a lot of 2:1 muxes and routing complexity.
    //m_axi_araddr[$clog2(HBM_PC_SIZE_IN_BYTES)+0] ^= pass_q;

    // Swap addresses with the read channel paired with this one.
    // This swap could be handled in point_to_ntt but using the
    // built in crossbar in the HBM IP to perform the swap saves
    // a lot of 2:1 muxes and routing complexity.
    //m_axi_araddr[$clog2(HBM_PC_SIZE_IN_BYTES)+1] ^= get_swap_from_beat(arbeat_q);

    ar_step        = m0_axi_arvalid && m0_axi_arready ||
                     m1_axi_arvalid && m1_axi_arready;

    r_step         = m0_axi_rvalid  && m0_axi_rready ||
                     m1_axi_rvalid  && m1_axi_rready;

    utilized       = r_step; // Just for debug.

    wrap           = beat_id_t'(LAST_BEAT);
  end

  assign dbg_rbeat = rbeat_q;
  assign dbg_rstep = r_step;
   
  // ----------------------------------------------------------------------
  // Next state logic.

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      pass_q    <= START_PASS;
      state_q   <= IDLE;
    end else begin
      pass_q    <= pass_d;
      state_q   <= state_d;
    end
  end

  always_comb begin
    pass_d       = pass_q;
    state_d      = state_q;
    restart      = 1'b0;

    done_o       = state_q inside {DONE};
    hold_o       = state_q inside {DONE, PASS};

    case (state_q)
      IDLE: begin
        if (start_i) begin
          restart  = 1'b1;
          state_d  = BUSY;
        end
      end
      BUSY: begin
        if (ar_done && r_done) begin
          state_d  = PASS;
        end
      end
      PASS: begin
        if (pass_q == LAST_PASS) begin
          restart  = 1'b1;
          state_d = DONE;
        end else if (release_i) begin
          pass_d   = 1'b1;
          restart  = 1'b1;
          state_d  = BUSY;
        end
      end
      DONE: begin
        if (release_i) begin
          state_d  = IDLE;
          pass_d   = 1'b0;
        end
      end
    endcase

    wdata_o   = pass_q ? m1_axi_rdata : m0_axi_rdata;
    we_o      = {NPPCH{r_step}};
    waddr_o   = {NPPCH{fine_t'(rbeat_q)}};
    wcoarse   = coarse_t'(rbeat_q >> $clog2(FIFO_FINE_DEPTH));
    volume    = coarse_t'(wcoarse - rcoarse_q);
    pause     = volume == FIFO_COARSE_DEPTH;
  end

endmodule
