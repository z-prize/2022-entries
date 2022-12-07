`default_nettype none

module bucket_storage #(
    N                                                               = 3,
    W                                                               = 32,
    D                                                               = 64*1024,
    M                                                               = 32,
    BUCK_PIPE_D                                                     = 6,
    D_L                                                             = $clog2(D)
)(
    input wire [64-1:0]                                             timestamp,
    input wire [64-1:0]                                             init_ts,
    input wire [N-1:0][W-1:0]                                       init_value,

    input wire  [D_L-1:0]                                           rd_addr,
    input wire  [M-1:0]                                             rd_meta_i,
    output logic[N-1:0][W-1:0]                                      rd_data,
    output logic[M-1:0]                                             rd_meta_o,

    input wire  [1-1:0]                                             wr_en,
    input wire  [D_L-1:0]                                           wr_addr,
    input wire  [N-1:0][W-1:0]                                      wr_data,

    input wire  [1-1:0]                                             clk,
    input wire  [1-1:0]                                             rst
);

bit [64-1:0]            rd_ts_i;

// logic [1-1:0] wr_en_i;
// logic [D_L-1:0] wr_addr_i;
// logic [N-1:0][W-1:0] wr_data_i;

// logic [N-1:0][W-1:0] rd_data_i;

// logic [BUCK_PIPE_D-1:0][M-1:0] meta_p;


// assign rd_meta_o = meta_p[BUCK_PIPE_D-1];

// always_ff@(posedge clk) begin
//     integer i;
//     meta_p[0] <= rd_meta_i;
//     for (i = 1; i < BUCK_PIPE_D; i ++)
//         meta_p[i] <= meta_p[i-1];
// end

// always_ff@(posedge clk) wr_en_i <= wr_en;
// always_ff@(posedge clk) wr_addr_i <= wr_addr;
// always_ff@(posedge clk) wr_data_i <= wr_data;

// generate
//     for (genvar g_i = 0; g_i < N; g_i ++) always_ff@(posedge clk) rd_data[g_i] <= (rd_ts_i > init_ts) ? rd_data_i[g_i] : init_value[g_i];
// endgenerate

// simple_dual_port_ram #(
//     .ADDRESS_WIDTH          (D_L),
//     .DATA_WIDTH             ($bits({wr_ts_i, wr_data_i})),
//     .REGISTER_OUTPUT        (BUCK_PIPE_D-2),
//     .CLOCKING_MODE          ("common_clock"),
//     .MEMORY_PRIMITIVE       ("ultra"),
//     .WRITE_MODE             ("read_first")
// ) buck_ram_inst (

//     .rd_clock               (clk),
//     .rd_en                  ('1),
//     .rd_address             (rd_addr),
//     .q                      ({rd_ts_i, rd_data_i}),

//     .wr_clock               (clk),
//     .wr_en                  (wr_en_i),
//     .wr_address             (wr_addr_i),
//     .wr_byteenable          ('1),
//     .data                   ({wr_ts_i, wr_data_i})
// );



localparam RAM_W = 72;
localparam RAM_D = 4*1024;

localparam TOTAL_W = (W*N) + $bits(timestamp);
localparam N_COLS = (TOTAL_W+RAM_W-1) / RAM_W;
localparam N_ROW = D / RAM_D;

localparam RAM_D_L = $clog2(RAM_D);
localparam N_ROW_L = $clog2(N_ROW);

localparam PDEPTH = 7;

logic [PDEPTH-1:0][M-1:0] m_o_p;

logic [N_COLS-1:0][RAM_W-1:0] c06_ram_rd_data;
logic [N-1:0][W-1:0] c06_rd_data_n;

assign rd_meta_o = m_o_p[PDEPTH-1];

always_ff@(posedge clk) begin
    integer i;
    m_o_p[0] <= rd_meta_i;
    for (i = 1; i < PDEPTH; i ++)
        m_o_p[i] <= m_o_p[i-1];
end


generate

    assign {rd_ts_i, c06_rd_data_n} = c06_ram_rd_data;

    for (genvar g_i = 0; g_i < N; g_i ++) always_ff@(posedge clk) rd_data[g_i] <= (rd_ts_i > init_ts) ? c06_rd_data_n[g_i] : init_value[g_i];


    if (1) begin: SLR

        logic [1-1:0]                   c01_wr_en;
        logic [D_L-1:0]                 c01_wr_addr;
        logic [N_COLS-1:0][RAM_W-1:0]   c01_wr_data_all;
        logic [D_L-1:0]                 c01_rd_addr;

        always_ff@(posedge clk) begin
            c01_wr_en                   <= wr_en;
            c01_wr_addr                 <= wr_addr;
            c01_wr_data_all             <= {timestamp, wr_data};

            c01_rd_addr                 <= rd_addr;
        end

        for (genvar g_c = 0; g_c < N_COLS; g_c ++) begin: G_C

            logic [1-1:0]                   c02_wr_en;
            logic [RAM_D_L-1:0]             c02_wr_i;
            logic [N_ROW_L-1:0]             c02_wr_row;
            logic [RAM_W-1:0]               c02_wr_data;

            logic [RAM_D_L-1:0]             c02_rd_i;
            logic [N_ROW_L-1:0]             c02_rd_row;
            logic [N_ROW_L-1:0]             c03_rd_row;
            logic [N_ROW_L-1:0]             c04_rd_row;
            logic [N_ROW_L-1:0]             c05_rd_row;
            logic [N_ROW-1:0][RAM_W-1:0]    c05_ram_rd_data;

            always_ff@(posedge clk) begin
                c02_wr_en                   <= c01_wr_en;
                c02_wr_i                    <= c01_wr_addr[0+:RAM_D_L];
                c02_wr_row                  <= c01_wr_addr >> RAM_D_L;
                c02_wr_data                 <= c01_wr_data_all[g_c];

                c02_rd_i                    <= c01_rd_addr[0+:RAM_D_L];
                c02_rd_row                  <= c01_rd_addr >> RAM_D_L;
                c03_rd_row                  <= c02_rd_row;
                c04_rd_row                  <= c03_rd_row;
                c05_rd_row                  <= c04_rd_row;

                c06_ram_rd_data[g_c]        <= c05_ram_rd_data[c05_rd_row];
            end

            for (genvar g_r = 0; g_r < N_ROW; g_r ++) begin: G_R

                logic [1-1:0]               c03_ram_wr_en;
                logic [RAM_D_L-1:0]         c03_ram_wr_addr;
                logic [RAM_W-1:0]           c03_ram_wr_data;

                logic [RAM_D_L-1:0]         c03_ram_rd_addr;

                always_ff@(posedge clk) begin
                    c03_ram_wr_en           <= c02_wr_en & (c02_wr_row == g_r);
                    c03_ram_wr_addr         <= c02_wr_i;
                    c03_ram_wr_data         <= c02_wr_data;

                    c03_ram_rd_addr         <= c02_rd_i;
                end

                simple_dual_port_ram #(
                    .ADDRESS_WIDTH          (RAM_D_L),
                    .DATA_WIDTH             (RAM_W),
                    .REGISTER_OUTPUT        (1),
                    .CLOCKING_MODE          ("common_clock"),
                    .MEMORY_PRIMITIVE       ("ultra"),
                    .WRITE_MODE             ("read_first")
                ) ram_inst (

                    .wr_clock               (clk),
                    .wr_en                  (c03_ram_wr_en),
                    .wr_address             (c03_ram_wr_addr),
                    .wr_byteenable          ('1),
                    .data                   (c03_ram_wr_data),

                    .rd_clock               (clk),
                    .rd_en                  ('1),
                    .rd_address             (c03_ram_rd_addr),
                    .q                      (c05_ram_rd_data[g_r])
                );
            end
        end
    end
endgenerate

endmodule

`default_nettype wire
