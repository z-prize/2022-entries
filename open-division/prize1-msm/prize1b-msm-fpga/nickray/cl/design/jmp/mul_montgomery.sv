`default_nettype none

(* keep_hierarchy = "yes" *) module mul_montgomery #(
    WI=378,WQ=377,WR=384,WM=48*8,L=3,T0=32'h07F7_FBBB,T1=32'h0000_005F,T2=32'h07F7_FBBB,
    M=1,

    // BLS12-377
    Q=384'h1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001,
    QP=384'hbfa5205feec82e3d22f80141806a3cec5b245b86cced7a1335ed1347970debffd1e94577a00000008508bfffffffffff,

    R_I=1,
    R_O=1
)(
    input wire                                                      clk,
    input wire                                                      rst,

    input wire [WI-1:0]                                             in0,
    input wire [WI-1:0]                                             in1,
    input wire [M-1:0]                                              m_i,
    output logic [M-1:0]                                            m_o,
    output logic [WI-1:0]                                           out0
);

// always_ff@(negedge clk) $display("%t: ------",$time);

logic [WM-1:0] in0_r;
logic [WM-1:0] in1_r;
logic [M-1:0] m_i_r;

generate

    if (R_I) begin
        always_ff@(posedge clk) in0_r <= in0;
        always_ff@(posedge clk) in1_r <= in1;
        always_ff@(posedge clk) m_i_r <= m_i;
    end else begin
        assign in0_r = in0;
        assign in1_r = in1;
        assign m_i_r = m_i;
    end


    if (1) begin

logic [2-1:0][WM-1:0]           m0_i;
logic [2*WM-1:0]                m0;
logic [3-1:0][1-1:0]            m0_c;
logic [3-1:0][WM*2-WR-1:0]      m0_h;

logic [2-1:0][WM-1:0]           m1_i;
logic [WM*2-1:0]                m1;

logic [2-1:0][WM-1:0]           m2_i;
logic [WM*2-1:0]                m2;

logic [2-1:0][WR-1:0]           a0_i;
logic [WR+1-1:0]                a0;

logic [2-1:0][WR-1:0]           a1_i;
logic [WR-1:0]                  a1;

logic [5-1:0][M-1:0]            m_o_p;

assign m0_i[0] = in0_r;
assign m0_i[1] = in1_r;

mul_wide #(
    .L      (L),
    .W      (WM),
    .T      (T0),
    .M      (M)
) m0_inst (
    .clk    (clk),
    .rst    (rst),
    .in0    (m0_i[0]),
    .in1    (m0_i[1]),
    .out0   (m0),
    .m_i    (m_i_r),
    .m_o    (m_o_p[0])
);

assign m0_h[0] = m0 >> WR;
assign m0_c[0] = |m0[0+:WR]; // we don't need the lower WR bits, only need to konw if they cause a carry in the add-0
assign m1_i[0] = m0[0 +: WR];
assign m1_i[1] = QP;

mul_wide #(
    .L      (L),
    .W      (WM),
    .T      (T1),
    .M      ($bits({m0_c[0], m0_h[0], m_o_p[0]}))
) m1_inst (
    .clk    (clk),
    .rst    (rst),
    .in0    (m1_i[0]),
    .in1    (m1_i[1]),
    .out0   (m1),
    .m_i    ({m0_c[0], m0_h[0], m_o_p[0]}),
    .m_o    ({m0_c[1], m0_h[1], m_o_p[1]})
);

assign m2_i[0] = m1[0 +: WR];
assign m2_i[1] = Q;

mul_wide #(
    .L      (L),
    .W      (WM),
    .T      (T2),
    .M      ($bits({m0_c[1], m0_h[1], m_o_p[1]}))
) m2_inst (
    .clk    (clk),
    .rst    (rst),
    .in0    (m2_i[0]),
    .in1    (m2_i[1]),
    .out0   (m2),
    .m_i    ({m0_c[1], m0_h[1], m_o_p[1]}),
    .m_o    ({m0_c[2], m0_h[2], m_o_p[2]})
);

assign a0_i[0] = m2 >> WR;
assign a0_i[1] = m0_h[2];

piped_adder #(
    .W      (WR),
    .R      (1),
    .C      (1),
    .M      (M)
) a0_inst (
    .clk    (clk),
    .rst    (rst),
    .cin0   (m0_c[2]),
    .in0    (a0_i[0]),
    .in1    (a0_i[1]),
    .out0   (a0[0+:WR]),
    .cout0  (a0[WR]),
    .m_i    (m_o_p[2]),
    .m_o    (m_o_p[3])
);

assign a1_i[0] = a0;
assign a1_i[1] = a0 >= Q ? -Q : '0;

piped_adder #(
    .W      (WR),
    .R      (1),
    .C      (1),
    .M      (M)
) a1_inst (
    .clk    (clk),
    .rst    (rst),
    .cin0   ('0),
    .in0    (a1_i[0]),
    .in1    (a1_i[1]),
    .out0   (a1),
    .cout0  (),
    .m_i    (m_o_p[3]),
    .m_o    (m_o_p[4])
);

if (R_O) begin
    always_ff@(posedge clk) out0 <= a1;
    always_ff@(posedge clk) m_o <= m_o_p[4];
end else begin
    assign out0 = a1;
    assign m_o = m_o_p[4];
end


// always_ff@(posedge clk)
// $display("%t:%m: %x %x %x",$time, in0_r, in1_r, m0);







































end


endgenerate

endmodule




























`default_nettype wire
