// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module point_dma_w_channel
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

   input  point_t [NPPCH-1:0]              rdata_i,
   output logic   [NPPCH-1:0]              re_o,
   output fine_t  [NPPCH-1:0]              raddr_o,
   input  coarse_t                         wcoarse_i,
   output coarse_t                         rcoarse_o,

   output beat_id_t                        dbg_wbeat,
   output logic 			   dbg_wstep,

   // Tool generated ports.
   output logic                            m0_axi_awvalid,
   input  logic                            m0_axi_awready,
   output logic [C_M_AXI_ADDR_WIDTH-1:0]   m0_axi_awaddr,
   output logic [8-1:0]                    m0_axi_awlen,
   output logic                            m0_axi_wvalid,
   input  logic                            m0_axi_wready,
   output logic [C_M_AXI_DATA_WIDTH-1:0]   m0_axi_wdata,
   output logic [C_M_AXI_DATA_WIDTH/8-1:0] m0_axi_wstrb,
   output logic                            m0_axi_wlast,
   input  logic                            m0_axi_bvalid,
   output logic                            m0_axi_bready,

   output logic                            m1_axi_awvalid,
   input  logic                            m1_axi_awready,
   output logic [C_M_AXI_ADDR_WIDTH-1:0]   m1_axi_awaddr,
   output logic [8-1:0]                    m1_axi_awlen,
   output logic                            m1_axi_wvalid,
   input  logic                            m1_axi_wready,
   output logic [C_M_AXI_DATA_WIDTH-1:0]   m1_axi_wdata,
   output logic [C_M_AXI_DATA_WIDTH/8-1:0] m1_axi_wstrb,
   output logic                            m1_axi_wlast,
   input  logic                            m1_axi_bvalid,
   output logic                            m1_axi_bready,

   input  logic [C_M_AXI_ADDR_WIDTH-1:0]   ctrl_addr_offset0,
   input  logic [C_M_AXI_ADDR_WIDTH-1:0]   ctrl_addr_offset1
   );

  import axi_hbm_pkg::*;

  localparam int LEN  = DMA_LEN;
  localparam int LANE = ID / (N_HBM_PC / NLANE);

  typedef enum logic [1:0] {IDLE, BUSY, PASS, DONE} state_e;

  typedef struct packed {
    logic               last;
    point_t [NPPCH-1:0] data;
  } timing_fifo_data_s;

  timing_fifo_data_s timing_fifo_wdata,  timing_fifo_rdata;
  logic              timing_fifo_wvalid, timing_fifo_rvalid;
  logic              timing_fifo_wready, timing_fifo_rready;

  state_e   state_q,  state_d;
  pass_id_t pass_q,   pass_d;
  beat_id_t awbeat_q;
  beat_id_t wbeat_q;
  beat_id_t bbeat_q;
  beat_id_t wrap;

  logic  restart;
  logic  utilized;
  logic  pause;
  logic  read;
  logic  wvalid_q, wvalid_d;
  logic  wready;
  fine_t raddr_next;
  addr_t addr;

  // ----------------------------------------------------------------------
  // Pointer synchronization and gray/bin conversion.

  coarse_t wcoarse_gray, wcoarse_d, wcoarse_q;
  coarse_t rcoarse_gray;
  coarse_t rcoarse;

  cdc_sync #(.WIDTH($bits(coarse_t)), .RESET(0)) _wcoarse_gray
    (.clk_i, .rst_ni, .i(wcoarse_i), .o(wcoarse_gray));

  gray_to_bin #(.WIDTH($bits(coarse_t))) _wcoarse_d
    (.i(wcoarse_gray), .o(wcoarse_d));

  always_ff @(posedge clk_i) begin
    wcoarse_q <= wcoarse_d;
    rcoarse_o <= rcoarse_gray;
  end

  bin_to_gray #(.WIDTH($bits(coarse_t))) _rcoarse_gray
    (.i(rcoarse), .o(rcoarse_gray));

  // ----------------------------------------------------------------------
  // AXI burst tracking.

  logic aw_step, aw_done;
  dma_counter #(.MAX(N_BEATS), .STEP(LEN), .START(START_BEAT)) _counter_aw
    (
     .clk_i,
     .rst_ni,
     .restart_i(restart),
     .step_i(aw_step),
     .done_o(aw_done),
     .count_o(awbeat_q),
     .flag_o(),
     .wrap_i(wrap)
     );

  logic w_step, w_done, w_last;
  dma_counter #(.MAX(N_BEATS), .STEP(1), .FLAG(LEN-1), .START(START_BEAT)) _counter_w
    (
     .clk_i,
     .rst_ni,
     .restart_i(restart),
     .step_i(w_step),
     .done_o(w_done),
     .count_o(wbeat_q),
     .flag_o(w_last),
     .wrap_i(wrap)
     );

  logic b_step, b_done;
  dma_counter #(.MAX(N_BEATS), .STEP(LEN), .START(START_BEAT)) _counter_b
    (
     .clk_i,
     .rst_ni,
     .restart_i(restart),
     .step_i(b_step),
     .done_o(b_done),
     .count_o(bbeat_q),
     .flag_o(),
     .wrap_i(wrap)
     );

  // ----------------------------------------------------------------------
  // AXI signal assignments.

  always_comb begin
    addr = get_addr_from_beat(awbeat_q, LANE, pass_q, POINT_WRITE);

    m0_axi_awvalid = (pass_q == 1'b1) && (state_q == BUSY) && !aw_done;
    m1_axi_awvalid = (pass_q == 1'b0) && (state_q == BUSY) && !aw_done;

    timing_fifo_wvalid = wvalid_q && !w_done && !pause;
    timing_fifo_wdata  = '{last: w_last, data: rdata_i};

    m0_axi_wstrb   = '1;
    m1_axi_wstrb   = '1;

    m0_axi_bready  = 1'b1;
    m1_axi_bready  = 1'b1;

    m0_axi_awlen   = LEN-1;
    m1_axi_awlen   = LEN-1;

    m0_axi_awaddr  = addr | $bits(m0_axi_awaddr)'(ctrl_addr_offset0);
    m1_axi_awaddr  = addr | $bits(m1_axi_awaddr)'(ctrl_addr_offset1);

    // Removed do to suspected HBM crossbar IP bug.
    // Trade addresses with the write channel paired with this one.
    // This uses the built in crossbar in the HBM IP to ping pong
    // between read and write buffers for first and second passes
    // thus saving a lot of 2:1 muxes and routing complexity.
    //m_axi_awaddr[$clog2(HBM_PC_SIZE_IN_BYTES)+0] ^= pass_q;

    // Swap addresses with the read channel paired with this one.
    // This swap could be handled in point_to_ntt but using the
    // built in crossbar in the HBM IP to perform the swap saves
    // a lot of 2:1 muxes and routing complexity.
    //m_axi_awaddr[$clog2(HBM_PC_SIZE_IN_BYTES)+1] ^= get_swap_from_beat(awbeat_q);

    aw_step        = m0_axi_awvalid && m0_axi_awready ||
                     m1_axi_awvalid && m1_axi_awready;

    w_step         = timing_fifo_wvalid && timing_fifo_wready;

    b_step         = m0_axi_bvalid  && m0_axi_bready ||
                     m1_axi_bvalid  && m1_axi_bready;

    utilized       = w_step; // Just for debug.

    wrap           = beat_id_t'(LAST_BEAT);
  end

  assign dbg_wbeat = wbeat_q;
  assign dbg_wstep = w_step;
   
  // ----------------------------------------------------------------------
  // Next state logic.

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      pass_q    <= START_PASS;
      state_q   <= IDLE;
      wvalid_q  <= '0;
    end else begin
      pass_q    <= pass_d;
      state_q   <= state_d;
      wvalid_q  <= wvalid_d;
    end
  end

  assign pause = wcoarse_q == rcoarse;

  always_comb begin
    pass_d       = pass_q;
    state_d      = state_q;
    wvalid_d     = wvalid_q;
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
        if (aw_done && w_done && b_done) begin
          state_d  = PASS;
        end
      end
      PASS: begin
        if (pass_q == LAST_PASS) begin
          restart  = 1'b1;
          state_d  = DONE;
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

    rcoarse   = coarse_t'(wbeat_q >> $clog2(FIFO_FINE_DEPTH));
    read      = (state_q == BUSY) && !w_done && !pause;

    case (pass_q)
      1'b0: wready = m1_axi_wready;
      1'b1: wready = m0_axi_wready;
    endcase

    if (restart)                          wvalid_d = 1'b0;
    else if (!wvalid_q && read)           wvalid_d = 1'b1;
    else if (wvalid_q && !read && wready) wvalid_d = 1'b0;

    if (w_step) begin
      raddr_next = fine_t'(wbeat_q + 1'b1);
    end else begin
      raddr_next = fine_t'(wbeat_q);
    end

    raddr_o = {NPPCH{raddr_next}};
    re_o    = {NPPCH{wvalid_d}};
  end

  // ----------------------------------------------------------------------
  // Break write data timing paths.

  fifo
    #(
      .DATA_t(timing_fifo_data_s),
      .DEPTH(DMA_W_FIFO_DEPTH),
      .SYNC_RESET(1)
      ) _timing_fifo
      (
       .clk_i,
       .rst_ni,
       .wdata_i(timing_fifo_wdata),
       .wvalid_i(timing_fifo_wvalid),
       .wready_o(timing_fifo_wready),
       .rdata_o(timing_fifo_rdata),
       .rvalid_o(timing_fifo_rvalid),
       .rready_i(timing_fifo_rready)
       );

  always_comb begin
    m0_axi_wdata  = timing_fifo_rdata.data;
    m1_axi_wdata  = timing_fifo_rdata.data;

    m0_axi_wlast  = timing_fifo_rdata.last;
    m1_axi_wlast  = timing_fifo_rdata.last;

    m0_axi_wvalid = timing_fifo_rvalid && (pass_q == 1'b1);
    m1_axi_wvalid = timing_fifo_rvalid && (pass_q == 1'b0);

    timing_fifo_rready = pass_q ? m0_axi_wready : m1_axi_wready;
  end

endmodule
