`default_nettype none

module showahead_fifo_nx1 #(
    N                   = 8,
    N_L                 = $clog2(N),
    WIDTH               = 1,
    DEPTH               = 512,
    D_L                 = $clog2(DEPTH),
    FULL_THRESH         = DEPTH-6
) (
   input wire wr_clk,
   input wire wr_req,
   input wire [N-1:0][WIDTH-1:0] wr_data,
   output wire wr_full,
   output wire wr_full_b,
   output wire [D_L+1-1:0] wr_count,

   input wire rd_clk,
   input wire rd_req,
   input wire rd_all,
   output wire [WIDTH-1:0] rd_data,
   output wire [N-1:0][WIDTH-1:0] rd_data_all,
   output wire rd_empty,
   output wire rd_not_empty,
   output wire [D_L+1-1:0] rd_count,

   input wire aclr
);

logic [N-1:0] full;

logic [N_L-1:0] i;
logic [N-1:0] out_vs;
logic [N-1:0][WIDTH-1:0] out_ds;
logic [N-1:0][D_L+1-1:0] wr_count_p;

assign wr_full = full[N-1];
assign wr_full_b = ~wr_full;
assign wr_count = wr_count_p[N-1];

assign rd_not_empty = out_vs[i];
assign rd_empty = ~rd_not_empty;
assign rd_data = out_ds[i];
assign rd_data_all = out_ds;

always_ff@(posedge rd_clk) begin
    if (rd_req)
        i <= rd_all ? 0 : (i + 1);
    if (aclr)
        i <= 0;
end

generate
    for (genvar g_i = 0; g_i < N; g_i ++) begin
        showahead_fifo #(
            .WIDTH                              (WIDTH),
            .DEPTH                              (DEPTH),
            .FULL_THRESH                        (FULL_THRESH)
        ) fifo_inst (
            .aclr                               (aclr),

            .wr_clk                             (wr_clk),
            .wr_req                             (wr_req),
            .wr_full                            (full[g_i]),
            .wr_data                            (wr_data[g_i]),
            .wr_count                           (wr_count_p[g_i]),

            .rd_clk                             (rd_clk),
            .rd_req                             (rd_req & ((i == g_i) | (rd_all & (i <= g_i)))),
            .rd_empty                           (),
            .rd_not_empty                       (out_vs[g_i]),
            .rd_count                           (),
            .rd_data                            (out_ds[g_i])
        );
    end
endgenerate

endmodule

`default_nettype wire
