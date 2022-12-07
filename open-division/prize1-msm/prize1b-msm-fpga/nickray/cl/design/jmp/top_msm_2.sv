`default_nettype none

module top_msm_2 #(
    DBG_WIDTH=1024,
    NO_DDR=4,
    DMA_N = 4
) (

    input wire [1-1:0]                                  avmm_read,
    input wire [1-1:0]                                  avmm_write,
    input wire [32-1:0]                                 avmm_address,
    input wire [32-1:0]                                 avmm_writedata,
    output logic [32-1:0]                               avmm_readdata,
    output logic [1-1:0]                                avmm_readdatavalid,
    output logic [1-1:0]                                avmm_waitrequest,

    input wire [2-1:0]                                  pcie_v,
    input wire [64-1:0]                                 pcie_a,
    input wire [2-1:0][256-1:0]                         pcie_d,

    output logic [1-1:0]                                dma_v,
    output logic [64-1:0]                               dma_a,
    input wire [1-1:0]                                  dma_f,
    output logic [DMA_N-1:0][14*32-1:0]                 dma_d,
    input wire [32-1:0]                                 dma_s,

    output logic [NO_DDR-1:0][1-1:0]                    ddr_rd_en,
    input wire   [NO_DDR-1:0][1-1:0]                    ddr_rd_pop,
    output logic [NO_DDR-1:0][64-1:0]                   ddr_rd_addr,
    output logic [NO_DDR-1:0][9-1:0]                    ddr_rd_sz,
    input wire   [NO_DDR-1:0][1-1:0]                    ddr_rd_v,
    input wire   [NO_DDR-1:0][512-1:0]                  ddr_rd_data,

    output logic [NO_DDR-1:0][1-1:0]                    ddr_wr_en,
    input wire   [NO_DDR-1:0][1-1:0]                    ddr_wr_pop,
    input wire   [NO_DDR-1:0][1-1:0]                    ddr_wr_res,
    output logic [NO_DDR-1:0][64-1:0]                   ddr_wr_addr,
    output wire  [NO_DDR-1:0][512-1:0]                  ddr_wr_data,

    output logic [DBG_WIDTH-1:0]                        dbg_wire,

    input wire clk,
    input wire rst
);

localparam CMD_START                    = 1;
localparam CMD_FINISH                   = 2;
localparam CMD_POINT                    = 3;
localparam CMD_SKIP                     = 4;

localparam P_W                          = 377;
localparam BUCK_PIPE_D                  = 7;
localparam PENDING_D                    = BUCK_PIPE_D+BUCK_PIPE_D+100;
localparam NO_POINTS_L                  = 26;
localparam POINT_BUFFER_SZ              = 512;
localparam POINT_BUFFER_SZ_L            = $clog2(POINT_BUFFER_SZ);
localparam NO_BUCKETS                   = 1<<16;
localparam NO_BUCKETS_L                 = $clog2(NO_BUCKETS);

typedef struct packed {
    logic [1-1:0] v;
    logic [4-1:0] t;
    logic [1-1:0] conflict;
    logic [1-1:0] asub;
    logic [NO_BUCKETS_L-1:0] buck_i;
    logic [1-1:0] buck_we;
} buck_meta_t;

logic [1-1:0]                           result_v;

logic [32-1:0]                          result_0_w;
logic [32-1:0]                          result_1_w;
logic [32-1:0]                          result_2_w;
logic [32-1:0]                          result_3_w;

logic [12-1:0][32-1:0]                  result_0;
logic [12-1:0][32-1:0]                  result_1;
logic [12-1:0][32-1:0]                  result_2;
logic [12-1:0][32-1:0]                  result_3;

logic [9-1:0]                           ddr_rd_len;

logic [64-1:0]                          timestamp = 0;
logic [64-1:0]                          timestamp_start;
logic [64-1:0]                          timestamp_finish;

logic [32-1:0]                          no_points;
logic [16-1:0]                          agg_buck_s;
logic [16-1:0]                          agg_buck_e;
logic [4-1:0][P_W-1:0]                  point_zero = 0;

logic [1-1:0]                           cmd_push;
logic [1-1:0]                           cmd_full;
logic [512-1:0]                         cmd_push_d;
logic [16-1:0]                          cmd_fifo_cnt;

logic [1-1:0]                           cmd_wr_v;
logic [64-1:0]                          cmd_wr_d;
logic [1-1:0]                           cmd_pop;
logic [1-1:0]                           cmd_pop_all;
logic [1-1:0]                           cmd_alone;
logic [1-1:0]                           cmd_v,          cmd_v_r;
logic [4-1:0]                           cmd_type,       cmd_type_r;
logic [1-1:0]                           cmd_asub,       cmd_asub_r;
logic [NO_BUCKETS_L-1:0]                cmd_buck_i,     cmd_buck_i_r;
logic [1-1:0]                           cmd_buck_we,    cmd_buck_we_r;

logic [3-1:0][2-1:0]                    st_ddr_rd;

logic [PENDING_D+2-1:0]                 c00_buck_conflicts;
buck_meta_t                             c00_buck_meta;

logic [PENDING_D+2-1:0]                 c01_buck_conflicts;
buck_meta_t                             c01_buck_meta;

buck_meta_t                             c02_buck_meta;

logic [3-1:0]                           st_buck;
logic [5-1:0]                           buck_wr_wait;
logic [1-1:0]                           buck_wr_en;
logic [NO_BUCKETS_L-1:0]                buck_wr_addr;
logic [4-1:0][P_W-1:0]                  buck_wr_data;
buck_meta_t                             buck_rd_meta_i;
buck_meta_t                             buck_rd_meta_o;
buck_meta_t                             buck_rd_meta_o_r;
logic [4-1:0][P_W-1:0]                  buck_rd_data;
logic [4-1:0][P_W-1:0]                  buck_rd_data_r;
logic [4-1:0][P_W-1:0]                  buck_agg_0;
logic [4-1:0][P_W-1:0]                  buck_agg_1;

buck_meta_t [PENDING_D-1:0]             pending_meta;

logic [1-1:0]                           point_wr_v;
logic [3-1:0]                           point_wr_f;
logic [3-1:0]                           point_rd_v;
logic [2-1:0]                           point_rd_p;
logic [3-1:0][P_W-1:0]                  point_rd_d;

logic [32-1:0]                          pipe_p_cnt;
logic [32-1:0]                          pipe_i_cnt;
logic [32-1:0]                          pipe_o_cnt;

logic [1-1:0]                           pipe_in_v;
logic [1-1:0]                           pipe_in_full;
logic [4-1:0]                           pipe_in_t;
logic [1-1:0]                           pipe_in_asub;
logic [NO_BUCKETS_L-1:0]                pipe_in_buck_i;
logic [1-1:0]                           pipe_in_buck_we;
logic [4-1:0][P_W-1:0]                  pipe_in0;
logic [4-1:0][P_W-1:0]                  pipe_in1;

logic [1-1:0]                           pipe_out_v;
logic [1-1:0]                           pipe_out_full;
logic [4-1:0]                           pipe_out_t;
logic [NO_BUCKETS_L-1:0]                pipe_out_buck_i;
logic [1-1:0]                           pipe_out_buck_we;
logic [4-1:0][P_W-1:0]                  pipe_out_point;
























//                AAA               VVVVVVVV           VVVVVVVVMMMMMMMM               MMMMMMMMMMMMMMMM               MMMMMMMM
//               A:::A              V::::::V           V::::::VM:::::::M             M:::::::MM:::::::M             M:::::::M
//              A:::::A             V::::::V           V::::::VM::::::::M           M::::::::MM::::::::M           M::::::::M
//             A:::::::A            V::::::V           V::::::VM:::::::::M         M:::::::::MM:::::::::M         M:::::::::M
//            A:::::::::A            V:::::V           V:::::V M::::::::::M       M::::::::::MM::::::::::M       M::::::::::M
//           A:::::A:::::A            V:::::V         V:::::V  M:::::::::::M     M:::::::::::MM:::::::::::M     M:::::::::::M
//          A:::::A A:::::A            V:::::V       V:::::V   M:::::::M::::M   M::::M:::::::MM:::::::M::::M   M::::M:::::::M
//         A:::::A   A:::::A            V:::::V     V:::::V    M::::::M M::::M M::::M M::::::MM::::::M M::::M M::::M M::::::M
//        A:::::A     A:::::A            V:::::V   V:::::V     M::::::M  M::::M::::M  M::::::MM::::::M  M::::M::::M  M::::::M
//       A:::::AAAAAAAAA:::::A            V:::::V V:::::V      M::::::M   M:::::::M   M::::::MM::::::M   M:::::::M   M::::::M
//      A:::::::::::::::::::::A            V:::::V:::::V       M::::::M    M:::::M    M::::::MM::::::M    M:::::M    M::::::M
//     A:::::AAAAAAAAAAAAA:::::A            V:::::::::V        M::::::M     MMMMM     M::::::MM::::::M     MMMMM     M::::::M
//    A:::::A             A:::::A            V:::::::V         M::::::M               M::::::MM::::::M               M::::::M
//   A:::::A               A:::::A            V:::::V          M::::::M               M::::::MM::::::M               M::::::M
//  A:::::A                 A:::::A            V:::V           M::::::M               M::::::MM::::::M               M::::::M
// AAAAAAA                   AAAAAAA            VVV            MMMMMMMM               MMMMMMMMMMMMMMMM               MMMMMMMM




logic [8-1:0] w_i;
logic [16-1:0][32-1:0] cnt = 0;

assign avmm_waitrequest = '0;

always_ff@(posedge clk) begin

    timestamp                           <= timestamp + 1;

    avmm_readdatavalid                  <= avmm_read;

    case (avmm_address[2+:8])
        8'h00: avmm_readdata            <= 32'h5000_0000;
        8'h01: avmm_readdata            <= 32'h0002_0002;

        8'h11: avmm_readdata            <= dma_s;

        8'h1e: avmm_readdata            <= dma_a[0 +:32];
        8'h1f: avmm_readdata            <= dma_a[32+:32];

        8'h20: avmm_readdata            <= cnt[w_i];
        8'h21: avmm_readdata            <= cmd_fifo_cnt;
        8'h22: avmm_readdata            <= timestamp_start[0*32+:32];
        8'h23: avmm_readdata            <= timestamp_start[1*32+:32];
        8'h24: avmm_readdata            <= timestamp_finish[0*32+:32];
        8'h25: avmm_readdata            <= timestamp_finish[1*32+:32];

        8'h30: avmm_readdata            <= result_v;
        8'h31: avmm_readdata            <= result_0_w;
        8'h32: avmm_readdata            <= result_1_w;
        8'h33: avmm_readdata            <= result_2_w;
        8'h34: avmm_readdata            <= result_3_w;
    endcase

    if (avmm_write) begin
    case (avmm_address[2+:8])

        8'h10: w_i                          <= avmm_writedata;
        8'h11: ddr_rd_len                   <= avmm_writedata;
        8'h1e: dma_a[0 +:32]                <= avmm_writedata;
        8'h1f: dma_a[32+:32]                <= avmm_writedata;

        8'h20: no_points                    <= avmm_writedata;
        8'h21: agg_buck_s                   <= avmm_writedata;
        8'h22: agg_buck_e                   <= avmm_writedata;
    endcase
    end else begin
    end
end

`define CNT(ci, expr) always_ff@(posedge clk) if (expr) cnt[ci] <= cnt[ci] + 1;

`CNT( 0, cmd_push & cmd_full)
`CNT( 1, cmd_v & (cmd_type == CMD_POINT) & (~(&point_rd_v)))
`CNT( 2, cmd_v & (|point_wr_f))
`CNT( 3, point_wr_v)
`CNT( 4, ddr_rd_v[0])
`CNT( 5, ddr_rd_v[1])
`CNT( 6, ddr_rd_v[2])
`CNT( 7, ddr_rd_en[0] & ddr_rd_pop[0])
`CNT( 8, ddr_rd_en[1] & ddr_rd_pop[1])
`CNT( 9, ddr_rd_en[2] & ddr_rd_pop[2])
`CNT(10, cmd_v_r)


always_ff@(posedge clk) begin
    if (cmd_v_r & (cmd_type_r == CMD_START))
        timestamp_start     <= timestamp;
    if (dma_v)
        timestamp_finish    <= timestamp;
end

always_ff@(posedge clk) result_0_w <= result_0[w_i];
always_ff@(posedge clk) result_1_w <= result_1[w_i];
always_ff@(posedge clk) result_2_w <= result_2[w_i];
always_ff@(posedge clk) result_3_w <= result_3[w_i];














// PPPPPPPPPPPPPPPPP                                      DDDDDDDDDDDDD        DDDDDDDDDDDDD        RRRRRRRRRRRRRRRRR   
// P::::::::::::::::P                   >>>>>>>           D::::::::::::DDD     D::::::::::::DDD     R::::::::::::::::R  
// P::::::PPPPPP:::::P                   >:::::>          D:::::::::::::::DD   D:::::::::::::::DD   R::::::RRRRRR:::::R 
// PP:::::P     P:::::P                   >:::::>         DDD:::::DDDDD:::::D  DDD:::::DDDDD:::::D  RR:::::R     R:::::R
//   P::::P     P:::::P                    >:::::>          D:::::D    D:::::D   D:::::D    D:::::D   R::::R     R:::::R
//   P::::P     P:::::P                     >:::::>         D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R
//   P::::PPPPPP:::::P                       >:::::>        D:::::D     D:::::D  D:::::D     D:::::D  R::::RRRRRR:::::R 
//   P:::::::::::::PP   ---------------       >:::::>       D:::::D     D:::::D  D:::::D     D:::::D  R:::::::::::::RR  
//   P::::PPPPPPPPP     -:::::::::::::-      >:::::>        D:::::D     D:::::D  D:::::D     D:::::D  R::::RRRRRR:::::R 
//   P::::P             ---------------     >:::::>         D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R
//   P::::P                                >:::::>          D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R
//   P::::P                               >:::::>           D:::::D    D:::::D   D:::::D    D:::::D   R::::R     R:::::R
// PP::::::PP                            >:::::>          DDD:::::DDDDD:::::D  DDD:::::DDDDD:::::D  RR:::::R     R:::::R
// P::::::::P                           >>>>>>>           D:::::::::::::::DD   D:::::::::::::::DD   R::::::R     R:::::R
// P::::::::P                                             D::::::::::::DDD     D::::::::::::DDD     R::::::R     R:::::R
// PPPPPPPPPP                                             DDDDDDDDDDDDD        DDDDDDDDDDDDD        RRRRRRRR     RRRRRRR

generate

    always_ff@(posedge clk) if (pcie_v[0] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 0)) point_zero[0][  0+:256] <= pcie_d[0];
    always_ff@(posedge clk) if (pcie_v[1] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 0)) point_zero[0][256+:121] <= pcie_d[1];
    always_ff@(posedge clk) if (pcie_v[0] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 1)) point_zero[1][  0+:256] <= pcie_d[0];
    always_ff@(posedge clk) if (pcie_v[1] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 1)) point_zero[1][256+:121] <= pcie_d[1];
    always_ff@(posedge clk) if (pcie_v[0] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 2)) point_zero[2][  0+:256] <= pcie_d[0];
    always_ff@(posedge clk) if (pcie_v[1] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 2)) point_zero[2][256+:121] <= pcie_d[1];
    always_ff@(posedge clk) if (pcie_v[0] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 3)) point_zero[3][  0+:256] <= pcie_d[0];
    always_ff@(posedge clk) if (pcie_v[1] & (pcie_a[32+:3] == 5) & (pcie_a[6+:2] == 3)) point_zero[3][256+:121] <= pcie_d[1];

    for (genvar g_i = 0; g_i < 3; g_i ++) begin: POINT_IN

        localparam logic [64-1:0] PCIE_OFF = {32'h0000_0001 + g_i, 32'h0000_0000};

        logic [1-1:0] pcie_iv;
        logic [64-1:0] pcie_ia;
        logic [512-1:0] pcie_id;

        logic [64-1:0] point_wr_a;

        logic [1-1:0] point_wr_m_v;
        logic [64-1:0] point_wr_m_a;
        logic [512-1:0] point_wr_m_d;

        pcie_inorder #(
            .W                          (512),
            .D                          (512),
            .REG_O(1),
            .ADDR_MASK                  (64'hFFFF_FFFF_0000_0000),
            .ADDR_VAL                   (PCIE_OFF)
        ) pcie_inorder_inst (
            .pcie_v                     (pcie_v),
            .pcie_a                     (pcie_a),
            .pcie_d                     (pcie_d),

            .out_v                      (pcie_iv),
            .out_p                      ('1),
            .out_a                      (pcie_ia),
            .out_d                      (pcie_id),
            .out_s                      (),

            .clk                        (clk),
            .rst                        (rst)
        );

        always_ff@(posedge clk) begin
            if (cmd_v_r & (cmd_type_r == CMD_START))
                point_wr_a                      <= (no_points << 6);
            else
            if (point_wr_v)
                point_wr_a                      <= point_wr_a + (1 << 6);
        end

        // pcie and point never happen at the same time
        always_ff@(posedge clk) begin
            point_wr_m_v <= point_wr_v | pcie_iv;
            point_wr_m_a <= point_wr_v ? point_wr_a : {32'h0, pcie_ia[0+:32]};
            point_wr_m_d <= point_wr_v ? {
                    buck_rd_meta_o_r.buck_i,
                    buck_rd_meta_o_r.buck_we,
                    buck_rd_meta_o_r.asub,
                    point_rd_d[g_i]
                } : pcie_id;
        end

        showahead_fifo #(
            .WIDTH                              ($bits({pcie_id, pcie_ia})),
            .FULL_THRESH                        (1024-64),
            .DEPTH                              (1024)
        ) ddr_wr_fifo_inst (
            .aclr                               (rst),

            .wr_clk                             (clk),
            .wr_req                             (point_wr_m_v),
            .wr_full                            (point_wr_f[g_i]),
            .wr_data                            ({point_wr_m_d, point_wr_m_a}),
            .wr_count                           (),

            .rd_clk                             (clk),
            .rd_req                             (ddr_wr_en[g_i] & ddr_wr_pop[g_i]),
            .rd_empty                           (),
            .rd_not_empty                       (ddr_wr_en[g_i]),
            .rd_count                           (),
            .rd_data                            ({ddr_wr_data[g_i], ddr_wr_addr[g_i]})
        );

        always_ff@(posedge clk)
        if (point_wr_m_v)
        $display("%t: %m.point_to_ddr: %x %x",$time, point_wr_m_a, point_wr_m_d[0+:32]);
    end
endgenerate

//         CCCCCCCCCCCCCMMMMMMMM               MMMMMMMMDDDDDDDDDDDDD                         FFFFFFFFFFFFFFFFFFFFFF
//      CCC::::::::::::CM:::::::M             M:::::::MD::::::::::::DDD                      F::::::::::::::::::::F
//    CC:::::::::::::::CM::::::::M           M::::::::MD:::::::::::::::DD                    F::::::::::::::::::::F
//   C:::::CCCCCCCC::::CM:::::::::M         M:::::::::MDDD:::::DDDDD:::::D                   FF::::::FFFFFFFFF::::F
//  C:::::C       CCCCCCM::::::::::M       M::::::::::M  D:::::D    D:::::D                    F:::::F       FFFFFF
// C:::::C              M:::::::::::M     M:::::::::::M  D:::::D     D:::::D                   F:::::F             
// C:::::C              M:::::::M::::M   M::::M:::::::M  D:::::D     D:::::D                   F::::::FFFFFFFFFF   
// C:::::C              M::::::M M::::M M::::M M::::::M  D:::::D     D:::::D ---------------   F:::::::::::::::F   
// C:::::C              M::::::M  M::::M::::M  M::::::M  D:::::D     D:::::D -:::::::::::::-   F:::::::::::::::F   
// C:::::C              M::::::M   M:::::::M   M::::::M  D:::::D     D:::::D ---------------   F::::::FFFFFFFFFF   
// C:::::C              M::::::M    M:::::M    M::::::M  D:::::D     D:::::D                   F:::::F             
//  C:::::C       CCCCCCM::::::M     MMMMM     M::::::M  D:::::D    D:::::D                    F:::::F             
//   C:::::CCCCCCCC::::CM::::::M               M::::::MDDD:::::DDDDD:::::D                   FF:::::::FF           
//    CC:::::::::::::::CM::::::M               M::::::MD:::::::::::::::DD                    F::::::::FF           
//      CCC::::::::::::CM::::::M               M::::::MD::::::::::::DDD                      F::::::::FF           
//         CCCCCCCCCCCCCMMMMMMMM               MMMMMMMMDDDDDDDDDDDDD                         FFFFFFFFFFF           

generate
    for (genvar g_i = 0; g_i < 1; g_i ++) begin: CMD_IN

        logic [1-1:0] cmd_pcie_v;
        logic [512-1:0] cmd_pcie_d;

        logic [64-1:0] cmd_d;

        logic [32-1:0]                                          cmd_wr_cnt;

        pcie_inorder #(
            .W(512),
            .D(512),
            .REG_O(1),
            .ADDR_MASK(64'hFFFF_FFFF_0000_0000),
            .ADDR_VAL(64'h0000_0004_0000_0000)
        ) pcie_inorder_inst (
            .pcie_v                     (pcie_v),
            .pcie_a                     (pcie_a),
            .pcie_d                     (pcie_d),

            .out_v                      (cmd_pcie_v),
            .out_p                      ('1),
            .out_a                      (),
            .out_d                      (cmd_pcie_d),
            .out_s                      (),

            .clk                        (clk),
            .rst                        (rst)
        );

        logic [1-1:0]                   asub;
        logic [1-1:0]                   wr_en;
        logic [NO_BUCKETS_L-1:0]        buck_i;

        assign {buck_i, wr_en, asub}    = ddr_rd_data[0][P_W+:NO_BUCKETS_L+2];

        always_ff@(posedge clk) begin

            if (cmd_v_r & (cmd_type_r == CMD_START)) begin
                cmd_wr_cnt                                      <= 0;
            end else begin
                // have to read all points, anything after that is a point and a command
                if (cmd_wr_cnt < no_points)
                    cmd_wr_cnt                                  <= cmd_wr_cnt + ddr_rd_v[0];
                else begin
                    cmd_wr_v                                    <= ddr_rd_v[0];
                    cmd_wr_d[0+:4]                              <= CMD_POINT;
                    cmd_wr_d[0+4+:1]                            <= asub;
                    cmd_wr_d[0+4+1+9+:NO_BUCKETS_L]             <= buck_i;
                    cmd_wr_d[0+4+1+9+16+:1]                     <= wr_en;
                    cmd_wr_d[64-1]                              <= 1'b1;
                end
            end

            if (rst) begin
                cmd_wr_v <= 0;
                cmd_wr_cnt <= 0;
            end
        end


        always_ff@(posedge clk) begin
            cmd_push <= cmd_wr_v | cmd_pcie_v;
            cmd_push_d <= cmd_wr_v ? cmd_wr_d : cmd_pcie_d;
        end

        showahead_fifo_nx1 #(
            .N                                  (8),
            .WIDTH                              (64),
            .FULL_THRESH                        (1024-6),
            .DEPTH                              (1024)
        ) cmd_fifo_inst (
            .aclr                               (rst),

            .wr_clk                             (clk),
            .wr_req                             (cmd_push),
            .wr_full                            (cmd_full),
            .wr_data                            ({cmd_push_d}),
            .wr_count                           (cmd_fifo_cnt),

            .rd_clk                             (clk),
            .rd_req                             (cmd_pop),
            .rd_all                             (cmd_pop_all),
            .rd_empty                           (),
            .rd_not_empty                       (cmd_v),
            .rd_count                           (),
            .rd_data_all                        (),
            .rd_data                            (cmd_d)
        );

        assign cmd_type                         = cmd_d[0+:4];
        assign cmd_asub                         = cmd_d[0+4+:1];
        assign cmd_buck_i                       = cmd_d[0+4+1+9+:16];
        assign cmd_buck_we                      = cmd_d[0+4+1+9+16+:1];
        assign cmd_alone                        = cmd_d[64-1];

        always_ff@(posedge clk)
        if (cmd_push)
        $display("%t: %m.cmd_push: %x %x - %x - %x %x",$time, cmd_wr_v, cmd_wr_cnt, cmd_pcie_v, cmd_push_d[0+:4], cmd_push_d[4+1+9+:16]);

    end
endgenerate

//         CCCCCCCCCCCCCMMMMMMMM               MMMMMMMMDDDDDDDDDDDDD        
//      CCC::::::::::::CM:::::::M             M:::::::MD::::::::::::DDD     
//    CC:::::::::::::::CM::::::::M           M::::::::MD:::::::::::::::DD   
//   C:::::CCCCCCCC::::CM:::::::::M         M:::::::::MDDD:::::DDDDD:::::D  
//  C:::::C       CCCCCCM::::::::::M       M::::::::::M  D:::::D    D:::::D 
// C:::::C              M:::::::::::M     M:::::::::::M  D:::::D     D:::::D
// C:::::C              M:::::::M::::M   M::::M:::::::M  D:::::D     D:::::D
// C:::::C              M::::::M M::::M M::::M M::::::M  D:::::D     D:::::D
// C:::::C              M::::::M  M::::M::::M  M::::::M  D:::::D     D:::::D
// C:::::C              M::::::M   M:::::::M   M::::::M  D:::::D     D:::::D
// C:::::C              M::::::M    M:::::M    M::::::M  D:::::D     D:::::D
//  C:::::C       CCCCCCM::::::M     MMMMM     M::::::M  D:::::D    D:::::D 
//   C:::::CCCCCCCC::::CM::::::M               M::::::MDDD:::::DDDDD:::::D  
//    CC:::::::::::::::CM::::::M               M::::::MD:::::::::::::::DD   
//      CCC::::::::::::CM::::::M               M::::::MD::::::::::::DDD     
//         CCCCCCCCCCCCCMMMMMMMM               MMMMMMMMDDDDDDDDDDDDD        

generate

    always_comb begin
        cmd_pop = cmd_v;
        cmd_pop_all = (cmd_type == 0) | (cmd_alone);

        // we cannot process a point command if
        // we run the risk of overflowing the 
        // write fifo to DDR in case there's conflicts
        if (cmd_type == CMD_POINT)
            cmd_pop = cmd_v & (~(|point_wr_f)) & (&point_rd_v);

        point_rd_p[0] = cmd_pop & (cmd_type == CMD_POINT);
    end

    always_ff@(posedge clk) begin

        cmd_v_r                                 <= cmd_v & cmd_pop;
        cmd_type_r                              <= cmd_type;

        // cmd coming back from DDR is already parsed
        if (cmd_alone) begin
            cmd_asub_r                          <= cmd_asub;
            cmd_buck_i_r                        <= cmd_buck_i;
            cmd_buck_we_r                       <= cmd_buck_we;
        end else begin
            cmd_asub_r                          <= cmd_buck_i[NO_BUCKETS_L-1];
            cmd_buck_i_r[NO_BUCKETS_L-1-1:0]    <= (cmd_buck_i[NO_BUCKETS_L-1] ? -cmd_buck_i : cmd_buck_i) - 1;
            cmd_buck_i_r[NO_BUCKETS_L-1]        <= 1'b0;
            cmd_buck_we_r                       <= |cmd_buck_i;
        end

        if (rst) begin
            cmd_v_r <= 0;
        end
    end

endgenerate

// DDDDDDDDDDDDD        DDDDDDDDDDDDD        RRRRRRRRRRRRRRRRR                                 PPPPPPPPPPPPPPPPP   
// D::::::::::::DDD     D::::::::::::DDD     R::::::::::::::::R                   >>>>>>>      P::::::::::::::::P  
// D:::::::::::::::DD   D:::::::::::::::DD   R::::::RRRRRR:::::R                   >:::::>     P::::::PPPPPP:::::P 
// DDD:::::DDDDD:::::D  DDD:::::DDDDD:::::D  RR:::::R     R:::::R                   >:::::>    PP:::::P     P:::::P
//   D:::::D    D:::::D   D:::::D    D:::::D   R::::R     R:::::R                    >:::::>     P::::P     P:::::P
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R                     >:::::>    P::::P     P:::::P
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::RRRRRR:::::R                       >:::::>   P::::PPPPPP:::::P 
//   D:::::D     D:::::D  D:::::D     D:::::D  R:::::::::::::RR   ---------------       >:::::>  P:::::::::::::PP  
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::RRRRRR:::::R  -:::::::::::::-      >:::::>   P::::PPPPPPPPP    
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R ---------------     >:::::>    P::::P            
//   D:::::D     D:::::D  D:::::D     D:::::D  R::::R     R:::::R                    >:::::>     P::::P            
//   D:::::D    D:::::D   D:::::D    D:::::D   R::::R     R:::::R                   >:::::>      P::::P            
// DDD:::::DDDDD:::::D  DDD:::::DDDDD:::::D  RR:::::R     R:::::R                  >:::::>     PP::::::PP          
// D:::::::::::::::DD   D:::::::::::::::DD   R::::::R     R:::::R                 >>>>>>>      P::::::::P          
// D::::::::::::DDD     D::::::::::::DDD     R::::::R     R:::::R                              P::::::::P          
// DDDDDDDDDDDDD        DDDDDDDDDDDDD        RRRRRRRR     RRRRRRR                              PPPPPPPPPP          

/*

    d       = 4x128 bits
    x/y/t   = 3x128 bits

     d0  d1  d2  d3  d0  d1  d2  d3
    ----====____~~~~----====____~~~~----====____~~~~----====____~~~~----====____~~~~----====____~~~~----====____~~~~----====____~~~~----====____~~~~
    ----====____----====____----====____----====____----====____----====____----====____----====____----====____
    ---===___---===___---===___---===___---===___---===___---===___---===___---===___---===___---===___---===___---===___---===___---===___---===___
    x0 y0 t0  x1 y1 t1
*/

generate

    for (genvar g_i = 0; g_i < 3; g_i ++) begin: POINT_RD

        logic [32-1:0] rd_sz;
        logic [32-1:0] fetch_cnt;
        logic [32-1:0] rd_total;

        logic [32-1:0] point_rd_cnt;
        logic [32-1:0] cmd_point_cnt;

        always_ff@(posedge clk) begin

            if (cmd_v_r & (cmd_type_r == CMD_START))
                point_rd_cnt                                <= '0;
            else
                point_rd_cnt                                <= point_rd_cnt + point_rd_p[1];

            // wait for all commands to arrive from the host
            // after that we can read all extra points speculatively
            if (cmd_point_cnt < no_points) begin
                cmd_point_cnt                               <= cmd_point_cnt + (cmd_v_r & (cmd_type_r == CMD_POINT));
                rd_sz                                       <= (fetch_cnt + ddr_rd_len) <= no_points ? ddr_rd_len : (no_points - fetch_cnt);
            end else begin
                rd_sz                                       <= (fetch_cnt + ddr_rd_len) <= rd_total ? ddr_rd_len : (rd_total - fetch_cnt);
            end

            if (rst | (cmd_v_r & (cmd_type_r == CMD_START)))
                cmd_point_cnt                               <= '0;
        end

        always_ff@(posedge clk) begin
            case (st_ddr_rd                         [g_i])
                0: begin
                    fetch_cnt                               <= 0;

                    ddr_rd_en                       [g_i]   <= 0;
                    ddr_rd_addr                     [g_i]   <= 0;

                    rd_total                                <= no_points;

                    if (cmd_v_r & (cmd_type_r == CMD_START)) begin
                        st_ddr_rd                   [g_i]   <= 1;
                    end
                end
                1: begin
                    rd_total                                <= rd_total + ddr_wr_res[g_i];

                    if (rd_sz == 0) begin

                        // wait for all points to be processed
                        if (pipe_p_cnt == no_points)
                            st_ddr_rd               [g_i]   <= 0;

                    end else
                    // if enough room in the buffer to fetch rd_sz
                    if ((point_rd_cnt + (POINT_BUFFER_SZ/2)) >= (fetch_cnt + rd_sz)) begin

                        fetch_cnt                           <= fetch_cnt + rd_sz;

                        ddr_rd_en                   [g_i]   <= 1;
                        ddr_rd_sz                   [g_i]   <= rd_sz;

                        st_ddr_rd                   [g_i]   <= 2;
                    end
                end
                2: begin
                    rd_total                                <= rd_total + ddr_wr_res[g_i];

                    if (ddr_rd_pop[g_i]) begin

                        ddr_rd_en                   [g_i]   <= 0;
                        ddr_rd_addr                 [g_i]   <= ddr_rd_addr      [g_i] + (ddr_rd_sz[g_i] << 6);

                        st_ddr_rd                   [g_i]   <= 1;
                    end
                end
            endcase

            if (rst)
                st_ddr_rd                           [g_i]   <= 0;
        end

        showahead_fifo #(
            .WIDTH                              (1),
            .FULL_THRESH                        (POINT_BUFFER_SZ-64),
            .DEPTH                              (POINT_BUFFER_SZ)
        ) v_fifo_inst (
            .aclr                               (rst),

            .wr_clk                             (clk),
            .wr_req                             (ddr_rd_v           [g_i]),
            .wr_full                            (),
            .wr_data                            ('0),
            .wr_count                           (),

            .rd_clk                             (clk),
            .rd_req                             (point_rd_p[0]),
            .rd_empty                           (),
            .rd_not_empty                       (point_rd_v         [g_i]),
            .rd_count                           (),
            .rd_data                            ()
        );
        showahead_fifo #(
            .WIDTH                              (512),
            .FULL_THRESH                        (POINT_BUFFER_SZ-64),
            .DEPTH                              (POINT_BUFFER_SZ)
        ) d_fifo_inst (
            .aclr                               (rst),

            .wr_clk                             (clk),
            .wr_req                             (ddr_rd_v           [g_i]),
            .wr_full                            (),
            .wr_data                            (ddr_rd_data        [g_i]),
            .wr_count                           (),

            .rd_clk                             (clk),
            .rd_req                             (point_rd_p[1]),
            .rd_empty                           (),
            .rd_not_empty                       (),
            .rd_count                           (),
            .rd_data                            (point_rd_d         [g_i])
        );

        always_ff@(posedge clk)
        if (st_ddr_rd[g_i] != 0)
        $display("%t: %m.ddr.rd: %x %x %x",$time, rd_total, rd_sz, fetch_cnt);


        always_ff@(posedge clk)
        if (ddr_rd_v[g_i])
        $display("%t: %m.ddr[%0d]->point = %x",$time, g_i, ddr_rd_data[g_i][0+:32]);

    end

endgenerate


// BBBBBBBBBBBBBBBBB   UUUUUUUU     UUUUUUUU        CCCCCCCCCCCCCKKKKKKKKK    KKKKKKK
// B::::::::::::::::B  U::::::U     U::::::U     CCC::::::::::::CK:::::::K    K:::::K
// B::::::BBBBBB:::::B U::::::U     U::::::U   CC:::::::::::::::CK:::::::K    K:::::K
// BB:::::B     B:::::BUU:::::U     U:::::UU  C:::::CCCCCCCC::::CK:::::::K   K::::::K
//   B::::B     B:::::B U:::::U     U:::::U  C:::::C       CCCCCCKK::::::K  K:::::KKK
//   B::::B     B:::::B U:::::D     D:::::U C:::::C                K:::::K K:::::K   
//   B::::BBBBBB:::::B  U:::::D     D:::::U C:::::C                K::::::K:::::K    
//   B:::::::::::::BB   U:::::D     D:::::U C:::::C                K:::::::::::K     
//   B::::BBBBBB:::::B  U:::::D     D:::::U C:::::C                K:::::::::::K     
//   B::::B     B:::::B U:::::D     D:::::U C:::::C                K::::::K:::::K    
//   B::::B     B:::::B U:::::D     D:::::U C:::::C                K:::::K K:::::K   
//   B::::B     B:::::B U::::::U   U::::::U  C:::::C       CCCCCCKK::::::K  K:::::KKK
// BB:::::BBBBBB::::::B U:::::::UUU:::::::U   C:::::CCCCCCCC::::CK:::::::K   K::::::K
// B:::::::::::::::::B   UU:::::::::::::UU     CC:::::::::::::::CK:::::::K    K:::::K
// B::::::::::::::::B      UU:::::::::UU         CCC::::::::::::CK:::::::K    K:::::K
// BBBBBBBBBBBBBBBBB         UUUUUUUUU              CCCCCCCCCCCCCKKKKKKKKK    KKKKKKK

generate

    for (genvar g_i = 0; g_i < 1; g_i ++) begin: G_I
        bucket_storage #(
            .N                                  (4),
            .W                                  (P_W),
            .D                                  (NO_BUCKETS>>1),
            .M                                  ($bits(buck_rd_meta_i)),
            .BUCK_PIPE_D                        (BUCK_PIPE_D)
        ) bucket_storage_inst (
            .timestamp                          (timestamp),
            .init_ts                            (timestamp_start),
            .init_value                         (point_zero),

            .rd_addr                            (buck_rd_meta_i.buck_i          [0+:NO_BUCKETS_L-1]),//[g_i]),
            .rd_meta_i                          (buck_rd_meta_i                 ),//[g_i]),
            .rd_data                            (buck_rd_data                   ),//[g_i]),
            .rd_meta_o                          (buck_rd_meta_o                 ),//[g_i]),

            .wr_en                              (buck_wr_en                     ),//[g_i]),
            .wr_addr                            (buck_wr_addr                   [0+:NO_BUCKETS_L-1]),//[g_i]),
            .wr_data                            (buck_wr_data                   ),//[g_i]),

            .clk                                (clk),
            .rst                                (rst)
        );
    end

    assign dma_d[0]                         = buck_agg_1[0];
    assign dma_d[1]                         = buck_agg_1[1];
    assign dma_d[2]                         = buck_agg_1[2];
    assign dma_d[3]                         = buck_agg_1[3];

    always_ff@(posedge clk) buck_wr_en      <= pipe_out_v & pipe_out_buck_we;
    always_ff@(posedge clk) buck_wr_addr    <= pipe_out_buck_i;
    always_ff@(posedge clk) buck_wr_data    <= pipe_out_point;

    always_comb begin
        integer i;
        for (i = 0; i < PENDING_D; i ++) begin
            c00_buck_conflicts[i]           = pending_meta[i].v & (cmd_buck_i_r == pending_meta[i].buck_i);
        end
        c00_buck_conflicts[PENDING_D+0]     = c01_buck_meta.v & (cmd_buck_i_r == c01_buck_meta.buck_i);
        c00_buck_conflicts[PENDING_D+1]     = c02_buck_meta.v & (cmd_buck_i_r == c02_buck_meta.buck_i);
    end

    always_ff@(posedge clk) buck_rd_data_r  <= buck_rd_data;
    always_ff@(posedge clk) buck_rd_meta_o_r<= buck_rd_meta_o;
    always_ff@(posedge clk) point_rd_p[1]   <= buck_rd_meta_o.v & (buck_rd_meta_o.t == CMD_POINT);
    always_ff@(posedge clk) point_wr_v      <= buck_rd_meta_o.v & (buck_rd_meta_o.t == CMD_POINT) & buck_rd_meta_o.conflict;

    always_ff@(posedge clk) begin
        integer i;

        case (st_buck)
            0: begin
                dma_v                       <= 0;

                // c00
                c01_buck_meta.v             <= cmd_v_r & (cmd_type_r == CMD_POINT);
                c01_buck_meta.t             <= cmd_type_r;
                c01_buck_meta.buck_i        <= cmd_buck_i_r;
                c01_buck_meta.buck_we       <= cmd_buck_we_r;
                c01_buck_meta.asub          <= cmd_asub_r;
                c01_buck_conflicts          <= c00_buck_conflicts;

                // c01
                c02_buck_meta               <= c01_buck_meta;
                c02_buck_meta.conflict      <= (c01_buck_meta.buck_we) & (|c01_buck_conflicts);

                // c02
                buck_rd_meta_i              <= c02_buck_meta;
                buck_rd_meta_i.v            <= c02_buck_meta.v & (c02_buck_meta.t == CMD_POINT);

                pending_meta[0]             <= c02_buck_meta;
                pending_meta[0].v           <= c02_buck_meta.v & (c02_buck_meta.t == CMD_POINT) & (~c02_buck_meta.conflict);
                for (i = 1; i < PENDING_D; i ++) begin
                    pending_meta[i]         <= pending_meta[i-1];
                end

                // c03 + BUCK_D + 1
                pipe_in_v                   <= point_rd_p[1] & (~buck_rd_meta_o_r.conflict);
                pipe_in_full                <= 0;
                pipe_in_t                   <= CMD_POINT;
                pipe_in_asub                <= buck_rd_meta_o_r.asub;
                pipe_in_buck_i              <= buck_rd_meta_o_r.buck_i;
                pipe_in_buck_we             <= buck_rd_meta_o_r.buck_we;
                pipe_p_cnt                  <= pipe_p_cnt + (point_rd_p[1] & ~buck_rd_meta_o_r.conflict);

                pipe_in0                    <= buck_rd_data_r;
                pipe_in1[0]                 <= point_rd_d[0];
                pipe_in1[1]                 <= point_rd_d[1];
                pipe_in1[2]                 <= point_rd_d[2];
                pipe_in1[3]                 <= point_rd_d[2];

                if (cmd_v_r) begin
                    case (cmd_type_r)
                        CMD_START: begin
                            result_v        <= 0;
                            pipe_p_cnt      <= 0;
                        end
                        CMD_FINISH: begin
                        end
                        CMD_POINT: begin
                        end
                    endcase
                end

                buck_wr_wait                <= BUCK_PIPE_D;

                if (pipe_o_cnt == no_points)
                    st_buck                 <= 1;
            end
            // wait for last bucket to finish writing
            1: begin
                buck_agg_0                  <= point_zero;
                buck_agg_1                  <= point_zero;

                buck_rd_meta_i.buck_i       <= agg_buck_s;

                buck_wr_wait                <= buck_wr_wait - 1;
                st_buck                     <= (|buck_wr_wait) ? 1 : 3;
            end

            // read bucket
            3: begin
                buck_rd_meta_i.v            <= 1;
                buck_rd_meta_i.t            <= CMD_FINISH;
                buck_rd_meta_i.asub         <= 0;
                buck_rd_meta_i.buck_we      <= 0;
                buck_rd_meta_i.conflict     <= 0;

                pipe_in_v                   <= 0;

                st_buck                     <= 4;
            end

            // A0 <= B + A0 (full pipe)
            4: begin
                buck_rd_meta_i.v            <= 0;

                pipe_in_v                   <= buck_rd_meta_o_r.v;
                pipe_in_full                <= 1;
                pipe_in_t                   <= CMD_FINISH;
                pipe_in_asub                <= 0;
                pipe_in_buck_i              <= buck_rd_meta_o_r.buck_i;
                pipe_in_buck_we             <= buck_rd_meta_o_r.buck_we;

                pipe_in0                    <= buck_rd_data_r;
                pipe_in1                    <= buck_agg_0;

                if (buck_rd_meta_o_r.v)
                    st_buck                 <= 5;
            end
            // A1 <= A0 + A1 (full pipe)
            5: begin
                buck_agg_0                  <= pipe_out_point;

                pipe_in_v                   <= pipe_out_v;
                pipe_in_full                <= 1;
                pipe_in_t                   <= CMD_FINISH;
                pipe_in_asub                <= 0;
                pipe_in_buck_i              <= buck_rd_meta_o_r.buck_i;

                pipe_in0                    <= pipe_out_point;
                pipe_in1                    <= buck_agg_1;

                if (pipe_out_v)
                    st_buck                 <= 6;
            end
            6: begin
                buck_agg_1                  <= pipe_out_point;

                pipe_in_v                   <= 0;

                if (pipe_out_v) begin

                    buck_rd_meta_i.buck_i   <= buck_rd_meta_i.buck_i - 1;

                    dma_v                   <= (buck_rd_meta_i.buck_i == agg_buck_e);
                    result_0                <= pipe_out_point[0];
                    result_1                <= pipe_out_point[1];
                    result_2                <= pipe_out_point[2];
                    result_3                <= pipe_out_point[3];

                    st_buck                 <= (buck_rd_meta_i.buck_i == agg_buck_e) ? 0 : 3;
                    result_v                <= 1;
                end
            end

        endcase

        if (rst) begin
            result_v <= 0;
            st_buck <= 0;
            buck_rd_meta_i.v <= 0;

            for (i = 0; i < PENDING_D; i ++)
                pending_meta[i].v <= 0;
        end
    end

    always_ff@(posedge clk)
    if (buck_rd_meta_i.v)
    $display("%t: %m.buck_rd_i: %x %x %x %x",$time
        , buck_rd_meta_i.t
        , buck_rd_meta_i.asub
        , buck_rd_meta_i.buck_i
        , buck_rd_meta_i.conflict
    );

    always_ff@(posedge clk)
    if (buck_rd_meta_o.v)
    $display("%t: %m.buck_rd_o: %x %x %x %x - %x %x %x %x - %x %x %x",$time
        , buck_rd_meta_o.t
        , buck_rd_meta_o.asub
        , buck_rd_meta_o.buck_i
        , buck_rd_meta_o.conflict
        , buck_rd_data[0][0+:32]
        , buck_rd_data[1][0+:32]
        , buck_rd_data[2][0+:32]
        , buck_rd_data[3][0+:32]
        , point_rd_d[0][0+:32]
        , point_rd_d[1][0+:32]
        , point_rd_d[2][0+:32]
    );

    always_ff@(posedge clk)
    if (buck_wr_en)
    $display("%t: %m.buck_wr: %x - %x %x %x %x",$time
        , buck_wr_addr
        , buck_wr_data[0][0+:32]
        , buck_wr_data[1][0+:32]
        , buck_wr_data[2][0+:32]
        , buck_wr_data[3][0+:32]
    );

endgenerate

// PPPPPPPPPPPPPPPPP   IIIIIIIIIIPPPPPPPPPPPPPPPPP   EEEEEEEEEEEEEEEEEEEEEE
// P::::::::::::::::P  I::::::::IP::::::::::::::::P  E::::::::::::::::::::E
// P::::::PPPPPP:::::P I::::::::IP::::::PPPPPP:::::P E::::::::::::::::::::E
// PP:::::P     P:::::PII::::::IIPP:::::P     P:::::PEE::::::EEEEEEEEE::::E
//   P::::P     P:::::P  I::::I    P::::P     P:::::P  E:::::E       EEEEEE
//   P::::P     P:::::P  I::::I    P::::P     P:::::P  E:::::E             
//   P::::PPPPPP:::::P   I::::I    P::::PPPPPP:::::P   E::::::EEEEEEEEEE   
//   P:::::::::::::PP    I::::I    P:::::::::::::PP    E:::::::::::::::E   
//   P::::PPPPPPPPP      I::::I    P::::PPPPPPPPP      E:::::::::::::::E   
//   P::::P              I::::I    P::::P              E::::::EEEEEEEEEE   
//   P::::P              I::::I    P::::P              E:::::E             
//   P::::P              I::::I    P::::P              E:::::E       EEEEEE
// PP::::::PP          II::::::IIPP::::::PP          EE::::::EEEEEEEE:::::E
// P::::::::P          I::::::::IP::::::::P          E::::::::::::::::::::E
// P::::::::P          I::::::::IP::::::::P          E::::::::::::::::::::E
// PPPPPPPPPP          IIIIIIIIIIPPPPPPPPPP          EEEEEEEEEEEEEEEEEEEEEE

generate

    logic [1-1:0] clk_core, rst_core;

    assign clk_core = clk;
    assign rst_core = rst;

    always_ff@(posedge clk) begin
        pipe_i_cnt <= pipe_i_cnt + pipe_in_v;
        pipe_o_cnt <= pipe_o_cnt + pipe_out_v;

        if (cmd_v_r & (cmd_type_r == CMD_START)) begin
            pipe_i_cnt <= 0;
            pipe_o_cnt <= 0;
        end
    end

    twisted_edwards_prek_full #(
        .S(0),
        .M($bits({pipe_in_buck_we, pipe_in_buck_i, pipe_in_t, pipe_in_full, pipe_in_v})),
        .W(378),.WQ(377),.WR(384),.WM(384),.L(3),.T0(32'h07F7_F999),.T1(32'h0000_005F),.T2(32'h07F7_F999), .D_M(35)
    ) twisted_edwards_prek_full_inst (
        .clk(clk_core),
        .rst(rst_core),

        .asub(pipe_in_asub),

        .in0_x(pipe_in0[0]),
        .in0_y(pipe_in0[1]),
        .in0_z(pipe_in0[2]),
        .in0_t(pipe_in0[3]),

        .in1_x(pipe_in1[0]),
        .in1_y(pipe_in1[1]),
        .in1_z(pipe_in1[2]),
        .in1_t(pipe_in1[3]),

        .out0_x(pipe_out_point[0]),
        .out0_y(pipe_out_point[1]),
        .out0_z(pipe_out_point[2]),
        .out0_t(pipe_out_point[3]),

        .m_i({pipe_in_buck_we, pipe_in_buck_i, pipe_in_t, pipe_in_full, pipe_in_v}),
        .m_o({pipe_out_buck_we, pipe_out_buck_i, pipe_out_t, pipe_out_full, pipe_out_v})
    );

    always_ff@(posedge clk)
    if (pipe_in_v)
    $display("%t: %m.pipe_in: %x %x %x %x %x - %x %x %x %x - %x %x %x %x",$time
        , pipe_in_full
        , pipe_in_t
        , pipe_in_asub
        , pipe_in_buck_i
        , pipe_in_buck_we

        , pipe_in0[0][0+:32]
        , pipe_in0[1][0+:32]
        , pipe_in0[2][0+:32]
        , pipe_in0[3][0+:32]

        , pipe_in1[0][0+:32]
        , pipe_in1[1][0+:32]
        , pipe_in1[2][0+:32]
        , pipe_in1[3][0+:32]
    );
    always_ff@(posedge clk)
    if (pipe_out_v)
    $display("%t: %m.pipe_out: %x %x - %x %x %x %x - %x %x %x %x",$time
        , pipe_i_cnt
        , pipe_o_cnt

        , pipe_out_full
        , pipe_out_t
        , pipe_out_buck_i
        , pipe_out_buck_we

        , pipe_out_point[0][0+:32]
        , pipe_out_point[1][0+:32]
        , pipe_out_point[2][0+:32]
        , pipe_out_point[3][0+:32]
    );

endgenerate











always_ff@(posedge clk) dbg_wire[DBG_WIDTH-1:2] <= {


/*logic [1-1:0]                     */              result_v,

// /*// logic [32-1:0]                 */              result_0_w,
// /*// logic [32-1:0]                 */              result_1_w,
// /*// logic [32-1:0]                 */              result_2_w,
// /*// logic [32-1:0]                 */              result_3_w,

// /*// logic [12-1:0][32-1:0]         */              result_0,
// /*// logic [12-1:0][32-1:0]         */              result_1,
// /*// logic [12-1:0][32-1:0]         */              result_2,
// /*// logic [12-1:0][32-1:0]         */              result_3,

/*logic [9-1:0]                     */              ddr_rd_len,

// /*// logic [64-1:0]                 */              timestamp,// = 0,
// /*// logic [64-1:0]                 */              timestamp_start,
// /*// logic [64-1:0]                 */              timestamp_finish,

/*logic [32-1:0]                    */              no_points,
// /*// logic [16-1:0]                 */              agg_buck_s,
// /*// logic [16-1:0]                 */              agg_buck_e,
// /*// logic [4-1:0][P_W-1:0]         */              point_zero = 0,

/*logic [1-1:0]                     */              cmd_push,
/*logic [1-1:0]                     */              cmd_full,
// /*// logic [512-1:0]                */              cmd_push_d,
/*logic [16-1:0]                    */              cmd_fifo_cnt,

/*logic [1-1:0]                     */              cmd_wr_v,
/*logic [64-1:0]                    */              cmd_wr_d,
/*logic [1-1:0]                     */              cmd_pop,
/*logic [1-1:0]                     */              cmd_pop_all,
/*logic [1-1:0]                     */              cmd_alone,
/*logic [1-1:0]                     */              cmd_v,          cmd_v_r,
/*logic [4-1:0]                     */              cmd_type,       cmd_type_r,
/*logic [1-1:0]                     */              cmd_asub,       cmd_asub_r,
/*logic [NO_BUCKETS_L-1:0]          */              cmd_buck_i,     cmd_buck_i_r,
/*logic [1-1:0]                     */              cmd_buck_we,    cmd_buck_we_r,

/*logic [3-1:0][2-1:0]              */              st_ddr_rd,

// /*// logic [PENDING_D+2-1:0]        */              c00_buck_conflicts,
// /*// buck_meta_t                    */              c00_buck_meta,

// /*// logic [PENDING_D+2-1:0]        */              c01_buck_conflicts,
// /*// buck_meta_t                    */              c01_buck_meta,

// /*// buck_meta_t                    */              c02_buck_meta,

/*logic [3-1:0]                     */              st_buck,
/*logic [5-1:0]                     */              buck_wr_wait,
/*logic [1-1:0]                     */              buck_wr_en,
/*logic [NO_BUCKETS_L-1:0]          */              buck_wr_addr,
// /*// logic [4-1:0][P_W-1:0]         */              buck_wr_data,
/*buck_meta_t                       */              buck_rd_meta_i,
/*buck_meta_t                       */              buck_rd_meta_o,
/*buck_meta_t                       */              buck_rd_meta_o_r,
// /*// logic [4-1:0][P_W-1:0]         */              buck_rd_data,
// /*// logic [4-1:0][P_W-1:0]         */              buck_rd_data_r,
// /*// logic [4-1:0][P_W-1:0]         */              buck_agg_0,
// /*// logic [4-1:0][P_W-1:0]         */              buck_agg_1,

// /*// buck_meta_t [PENDING_D-1:0]    */              pending_meta,

/*logic [1-1:0]                     */              point_wr_v,
/*logic [3-1:0]                     */              point_wr_f,
/*logic [3-1:0]                     */              point_rd_v,
/*logic [2-1:0]                     */              point_rd_p,
// /*// logic [3-1:0][P_W-1:0]         */              point_rd_d,

/*logic [32-1:0]                    */              pipe_p_cnt,
/*logic [32-1:0]                    */              pipe_i_cnt,
/*logic [32-1:0]                    */              pipe_o_cnt,

/*logic [1-1:0]                     */              pipe_in_v,
/*logic [1-1:0]                     */              pipe_in_full,
/*logic [4-1:0]                     */              pipe_in_t,
/*logic [1-1:0]                     */              pipe_in_asub,
/*logic [NO_BUCKETS_L-1:0]          */              pipe_in_buck_i,
/*logic [1-1:0]                     */              pipe_in_buck_we,

// /*// logic [4-1:0][P_W-1:0]         */              pipe_in0,
// /*// logic [4-1:0][P_W-1:0]         */              pipe_in1,

/*logic [1-1:0]                     */              pipe_out_v,
/*logic [1-1:0]                     */              pipe_out_full,
/*logic [4-1:0]                     */              pipe_out_t,
/*logic [NO_BUCKETS_L-1:0]          */              pipe_out_buck_i,
/*logic [1-1:0]                     */              pipe_out_buck_we,
// /*// logic [4-1:0][P_W-1:0]         */              pipe_out_point,

/*        logic [32-1:0]            */              POINT_RD[0].rd_sz,
/*        logic [32-1:0]            */              POINT_RD[0].fetch_cnt,
/*        logic [32-1:0]            */              POINT_RD[0].rd_total,

/*        logic [32-1:0]            */              POINT_RD[0].point_rd_cnt,

/*        logic [32-1:0]            */              POINT_RD[1].rd_sz,
/*        logic [32-1:0]            */              POINT_RD[1].fetch_cnt,
/*        logic [32-1:0]            */              POINT_RD[1].rd_total,

/*        logic [32-1:0]            */              POINT_RD[1].point_rd_cnt,

/*        logic [32-1:0]            */              POINT_RD[2].rd_sz,
/*        logic [32-1:0]            */              POINT_RD[2].fetch_cnt,
/*        logic [32-1:0]            */              POINT_RD[2].rd_total,

/*        logic [32-1:0]            */              POINT_RD[2].point_rd_cnt,

    |{
        pcie_v,
        cmd_v,
        dma_v
    }
};

assign dbg_wire[0+:2] = {
    rst,
    clk
};


always_ff@(posedge clk)
if (cmd_v)
$display("%t: %m.cmd: %x %x - %x %x %x",$time
    , cmd_pop
    , cmd_pop_all
    , cmd_type
    , cmd_asub
    , cmd_buck_i
);

always_ff@(posedge clk)
if (cmd_v_r)
$display("%t: %m.cmd_r: %x %x %x %x",$time
    , cmd_type_r
    , cmd_asub_r
    , cmd_buck_i_r
    , cmd_buck_we_r
);

always_ff@(posedge clk)
if (dma_v)
$display("%t: %m.dma: %x %x - %x %x %x %x", $time
    , dma_f
    , dma_s
    , dma_d[0]
    , dma_d[1]
    , dma_d[2]
    , dma_d[3]
);

always_ff@(posedge clk)
if (cmd_v_r | c01_buck_meta.v | c02_buck_meta.v)
$display("%t: %m.buck_conf: %x %x %x %b - %x %x %x %b - %x %x %x %x", $time
    , cmd_v_r
    , cmd_type_r
    , cmd_buck_i_r
    , c00_buck_conflicts

    , c01_buck_meta.v
    , c01_buck_meta.t
    , c01_buck_meta.buck_i
    , c01_buck_conflicts

    , c02_buck_meta.v
    , c02_buck_meta.t
    , c02_buck_meta.buck_i
    , c02_buck_meta.conflict
);

// always_ff@(posedge clk)
// if (|pcie_v)
// $display("%t: %m.pcie: %x %x",$time
//     , pcie_a
//     , pcie_d
// );

always_ff@(negedge clk)
$display("%t: -----------",$time);


endmodule


`default_nettype wire
