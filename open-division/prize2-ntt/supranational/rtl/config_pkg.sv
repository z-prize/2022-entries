// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

package config_pkg;

  import axi_hbm_pkg::addr_t;
  import axi_hbm_pkg::DATA_WIDTH_IN_BYTES;
  import axi_hbm_pkg::HBM_PC_SIZE_IN_BYTES;

  // ----------------------------------------------------------------------
  // Compile-time switches.
  
`ifdef LOOPBACK
  localparam bit LOOPBACK = `LOOPBACK;
`else
  localparam bit LOOPBACK = 0;
`endif

`ifdef ASYNC
  localparam bit ASYNC = `ASYNC;
`else
  localparam bit ASYNC = 0;
`endif

`ifdef DF
  localparam bit NOP = `DF
`else
  localparam bit NOP = 0;
`endif

`ifdef START_PASS
  localparam bit START_PASS = `START_PASS
`else
  localparam bit START_PASS = 0;
`endif

`ifdef SHIFT_PIPE_DEPTH
  localparam SHIFT_PIPE_DEPTH = `SHIFT_PIPE_DEPTH;
`else
  localparam SHIFT_PIPE_DEPTH = 4;
`endif

  // ----------------------------------------------------------------------
  // Point and memory based datatypes.

  typedef logic [63:0] point_t;
  typedef bit   [63:0] bit_point_t;

  // Given.
  localparam int K                   = 24;

  // Hardware single pass number of points.
  // Fixed but make variable?
  localparam int NLEVEL              = 12;
  localparam int NLEVEL0             = NLEVEL-7;

  // Very access pattern specific parameter
  // related to log2 of NLANE/2 and DMA_LEN.
  // This allows the access patterns of
  // pass 0 and pass 1 to work on a 64-point
  // tile basis.
  // Burst length for point DMA.
  localparam int DMA_LEN             = 8;
  localparam int GROUP               = 32;

  // Prime for this field.
  localparam point_t M               = 64'hffff_ffff_0000_0001;

  // Depth determined by the point DMA requirements.
  localparam int FIFO_COARSE_DEPTH   = 2;
  localparam int FIFO_FINE_DEPTH     = DMA_LEN*GROUP;

  // Timing cleanup FIFO.  Can be made transparent
  // by setting depth to zero at the potential expense
  // of timing.
  localparam int DMA_W_FIFO_DEPTH    = 2;

  // Calculated.
  localparam int N                   = 2**K;
  localparam int NLANE               = 16;
  localparam int NCH                 = NLANE*2;
  localparam int CHANNEL_WIDTH       = 8*DATA_WIDTH_IN_BYTES;
  localparam int NPPCH               = CHANNEL_WIDTH / $bits(point_t);
  localparam int KPPCH               = $clog2(NPPCH);
  localparam int PPCH_STRIDE         = 2**(KPPCH / 2);
  localparam int NBPP                = $bits(point_t) / 8;

  localparam int N_PASSES            = 2;
  localparam int K_PASSES            = $clog2(N_PASSES);
  localparam int LAST_PASS           = N_PASSES-1;

  localparam int N_CYCLES            = N / (NLANE*2);
  localparam int START_CYCLE         = 0;
  localparam int LAST_CYCLE          = N_CYCLES-1;

  localparam int K_BEATS             = K-($clog2(NLANE)+KPPCH);
  localparam int N_BEATS             = 2**K_BEATS;
  localparam int START_BEAT          = START_CYCLE/(N_CYCLES/N_BEATS);
  localparam int LAST_BEAT           = LAST_CYCLE/(N_CYCLES/N_BEATS);

  typedef logic   [K-1:0]             point_id_t;
  typedef logic   [KPPCH-1:0]         ppch_id_t;
  typedef logic   [$clog2(NLANE)-1:0] lane_id_t;
  typedef logic   [$clog2(NCH)-1:0]   channel_id_t;
  typedef logic   [K_BEATS-1:0]       beat_id_t;
  typedef point_t [NPPCH-1:0]         beat_data_t;
  typedef logic   [K_PASSES-1:0]      pass_id_t;

  typedef logic [$clog2(N_CYCLES)-1:0] cycle_id_t;

  typedef enum logic { POINT_WRITE, POINT_READ } point_op_e;

  typedef enum logic { NTT18, NTT24 } ntt_size_e;

  typedef logic [K-1:0] mem_index_t;

  typedef logic [$clog2(NLANE/2)-1:0] shift_t;

  typedef logic [$clog2(FIFO_COARSE_DEPTH*FIFO_FINE_DEPTH)-1:0] fine_t;
  typedef logic [$clog2(FIFO_COARSE_DEPTH):0]                   coarse_t;

  // ----------------------------------------------------------------------
  // Testbench index getters.

  function automatic addr_t idx_to_addr(input int idx);
    addr_t addr;
    addr = addr_t'(idx << $clog2(NBPP));
    return addr;
  endfunction

  function automatic int addr_to_idx(input addr_t addr);
    int idx;
    idx = int'(addr >> $clog2(NBPP));
    return idx;
  endfunction

  function automatic addr_t bank_optimize_addr(input addr_t addr);
    addr = {addr[22:14],
            addr[17:15] ^ addr[13:11],
            addr[10:0]};
    return addr;
  endfunction

  function automatic int get_channel_from_point;
    input int p;

    int ch;

    ch  = (p >>  1) % (NLANE/2);
    ch += (p >> 13) % (NLANE/2);
    ch %= NLANE/2;

    ch *= 2;
    ch += ((p >> 23) ^ (p >> 11)) & 1;

    return ch;
  endfunction

  function automatic int get_index_from_point;
    input int p;
    input bit last;

    bit_point_t p_bits;
    int idx;
    addr_t addr;

    p_bits = p;

    if (last) begin
      idx = {p_bits[17:16],
             p_bits[11:4],
             p_bits[22:18],
             p_bits[3:0],
             p_bits[12]};
    end else begin
      idx = {p_bits[17:16],
             p_bits[11:4],
             p_bits[22:18],
             p_bits[15:12],
             p_bits[0]};
    end

    addr = idx_to_addr(idx);
    addr = bank_optimize_addr(addr);
    idx  = addr_to_idx(addr);

    assert(idx < N/NLANE);
    return idx;
  endfunction

  function automatic mem_index_t tb_mem_index;
    input addr_t addr;

    mem_index_t result;
    // Should clip off unneeded upper bits.
    result = { addr[$bits(addr_t)-1:$clog2(HBM_PC_SIZE_IN_BYTES)+1],//+1 for read vs. write channel
               addr[$clog2(NBPP)+:(K-$clog2(NLANE))] };

    return result;
  endfunction

  function automatic int ntt_order_from_natural;
    input int p;
    input int nhw;
    input int pass;

    int zz_i, zz_j, zz_k, zz_l;
    int zz_ijk;
    int zz_ijkl;
    int nlevel, nlevel0;

    nlevel  = $clog2(nhw);
    nlevel0 = $clog2(nhw)-7;

    zz_i = p / 2 / NLANE % (nhw/2) / (1 << nlevel0);
    zz_j = p / 2 / NLANE % (nhw/2) % (1 << nlevel0);
    
    zz_k = p % 2;
    zz_l = p / 2 % NLANE + p / nhw / NLANE * NLANE;

    zz_ijk = zz_k * (nhw/2) + zz_j * (1<<(nlevel-nlevel0-1)) + zz_i;

    case (pass)
      0: zz_ijkl = zz_ijk * nhw + zz_l;
      1: zz_ijkl = zz_ijk + zz_l * nhw;
    endcase

    return zz_ijkl;
  endfunction

  // ----------------------------------------------------------------------
  // Helper functions.

  function automatic addr_t get_addr_from_beat;
    input beat_id_t    beat_id;
    input lane_id_t    lane_id;
    input pass_id_t    pass_id;
    input logic        is_rd;

    addr_t addr;

    case ({pass_id, is_rd})
      2'b01,
      2'b00,
      2'b10: begin
        addr = addr_t'({
                        beat_id[$clog2(DMA_LEN*GROUP)+0 +: 2],
                        beat_id[$clog2(DMA_LEN*GROUP)+2 +: 8],
                        beat_id[$clog2(DMA_LEN) +: $clog2(GROUP)],
                        $clog2(DMA_LEN)'(0),
                        $clog2(DATA_WIDTH_IN_BYTES)'(0)
                        });
      end
      2'b11: begin
        addr = addr_t'({
                        beat_id[$clog2(DMA_LEN*GROUP)+2 +: 2],
                        beat_id[$clog2(DMA_LEN*GROUP)+9] ^ lane_id[0],
                        beat_id[$clog2(DMA_LEN) +: $clog2(GROUP)],
                        beat_id[$clog2(DMA_LEN*GROUP)+0 +: 2],
                        beat_id[$clog2(DMA_LEN*GROUP)+4 +: 5],
                        $clog2(DMA_LEN)'(0),
                        $clog2(DATA_WIDTH_IN_BYTES)'(0)
                        });
      end
    endcase

    return bank_optimize_addr(addr);
  endfunction

  function automatic shift_t get_shift_from_cycle;
    input cycle_id_t   cycle_id;
    input logic        is_rd;

    shift_t shift;

    if (is_rd) begin
      shift = cycle_id[6 +: $bits(shift_t)];
    end else begin
      shift = $bits(shift_t)'('0) - cycle_id[6 +: $bits(shift_t)];
    end

    return shift;
  endfunction

  function automatic logic get_swap_from_beat;
    input beat_id_t   beat_id;

    logic swap;

    swap = beat_id[$clog2(N_BEATS)-1];

    return swap;
  endfunction

  function automatic logic get_swap_from_cycle;
    input cycle_id_t  cycle_id;

    logic swap;

    swap = cycle_id[$clog2(N_CYCLES)-1];

    return swap;
  endfunction

  function automatic ppch_id_t get_ppch_from_cycle;
    input cycle_id_t   cycle_id;
    input logic        pair;
    input pass_id_t    pass_id;

    ppch_id_t ppch_id;

    case (pass_id)
      1'b0: begin
        ppch_id[0] = pair;
        ppch_id[1] = cycle_id[$clog2(GROUP)];
      end
      1'b1: begin
        ppch_id[0] = cycle_id[$clog2(GROUP)];
        ppch_id[1] = pair;
      end
    endcase

    return ppch_id;
  endfunction

  function automatic fine_t get_fine_from_cycle;
    input cycle_id_t   cycle_id;
    input lane_id_t    lane_id;
    input pass_id_t    pass_id;

    logic [$clog2(DMA_LEN)-1:0] lsbs;
    fine_t addr;

    // Grab DMA_LEN's worth of lsbs and do the transform below
    // to grab the appropriate data when reading FIFO for
    // DMA to NTT and writing for NTT to DMA.  Only for
    // second pass.
    // CH 0,1: adjust factor = 7
    //   Original: 07654321
    //   Adjusted: 76543210
    //   Flipped:  01234567
    // CH 2,3: adjust factor = 6
    //   Original: 10765432
    //   Adjusted: 76543210
    //   Flipped:  01234567
    // ...
    // CH 14,15: adjust factor = 0
    //   Original: 76543210
    //   Adjusted: 76543210
    //   Flipped:  01234567
    //
    // ((NCH//2)-1)-(x+((NCH//2)-(ch//2)-1))
    // ((ch//2)-x) % (NCH//2)
    lsbs = cycle_id[$clog2(GROUP*N_CYCLES/N_BEATS)+:$clog2(DMA_LEN)];

    if (pass_id == 1'b1) begin
      lsbs = $clog2(DMA_LEN)'((lane_id>>1)-lsbs);
    end

    // Fine FIFO address = {group, burst}
    addr = fine_t'({cycle_id[$bits(cycle_id_t)-1:$clog2(DMA_LEN*GROUP*N_CYCLES/N_BEATS)],
                    cycle_id[0+:$clog2(GROUP)],
                    lsbs
                    });

    return addr;
  endfunction

endpackage
