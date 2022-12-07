`default_nettype none

module piped_adder #(
    W=384,
    C=1,
    M=1,
    R=0
)(
    input wire                                                      clk,
    input wire                                                      rst,

    input wire [1-1:0]                                              cin0,
    input wire [W-1:0]                                              in0,
    input wire [W-1:0]                                              in1,
    input wire [M-1:0]                                              m_i,
    output logic [M-1:0]                                            m_o,
    output logic [W-1:0]                                            out0,
    output logic [1-1:0]                                            cout0
);

generate
    if (C == 0) begin
        logic [W+1-1:0] s;
        assign s = in0 + in1 + cin0;
        if (R == 0) begin
            assign out0 = s[0+:W];
            assign cout0 = s[W];
            assign m_o = m_i;
        end else begin
            always_ff@(posedge clk) out0 <= s[0+:W];
            always_ff@(posedge clk) cout0 <= s[W];
            always_ff@(posedge clk) m_o <= m_i;
        end
    // don't split, just pipe
    // end else if (W < 192) begin
    end else if (0) begin

        localparam D = 1<<C;

        logic [D-1:0][M-1:0] m_o_p;
        logic [D-1:0][W-1:0] s;
        logic [D-1:0][1-1:0] c;

        assign out0 = s[D-1];
        assign cout0 = c[D-1];
        assign m_o = m_o_p[D-1];

        always_ff@(posedge clk) begin
            integer i;
            for (i = 1; i < D; i ++) begin
                s[i] <= s[i-1];
                c[i] <= c[i-1];
                m_o_p[i] <= m_o_p[i-1];
            end
        end

        piped_adder #(
            .W(W),
            .C(0),
            .R(R),
            .M(M)
        ) piped_adder_inst_0 (
            .clk                                                    (clk),
            .rst                                                    (rst),

            .cin0                                                   (cin0),
            .in0                                                    (in0),
            .in1                                                    (in1),
            .out0                                                   (s[0]),
            .cout0                                                  (c[0]),
            .m_i                                                    ({m_i}),
            .m_o                                                    ({m_o_p[0]})
        );

    end else begin
        localparam W1 = W / 2;
        localparam W2 = W - W1;

        logic [2-1:0] c;
        logic [W1-1:0] i00;
        logic [W2-1:0] i01;
        logic [W1-1:0] i10;
        logic [W2-1:0] i11;
        logic [W2-1:0] i0_p;
        logic [W2-1:0] i1_p;
        logic [W1-1:0] s0;
        logic [W1-1:0] s1;
        logic [W2-1:0] s2;
        logic [2-1:0][M-1:0] m_o_p;

        assign {i01, i00} = in0;
        assign {i11, i10} = in1;

        assign out0 = {s2, s1};
        assign m_o = m_o_p[1];
        assign cout0 = c[1];

        piped_adder #(
            .W(W1),
            .C(C-1),
            .R(1),
            .M(M+W2+W2)
        ) piped_adder_inst_0 (
            .clk                                                    (clk),
            .rst                                                    (rst),

            .cin0                                                   (cin0),
            .in0                                                    (i00),
            .in1                                                    (i10),
            .out0                                                   (s0),
            .cout0                                                  (c[0]),
            .m_i                                                    ({m_i, i11, i01}),
            .m_o                                                    ({m_o_p[0], i1_p, i0_p})
        );

        piped_adder #(
            .W(W2),
            .C(C-1),
            .R(1),
            .M(M+W1)
        ) piped_adder_inst_1 (
            .clk                                                    (clk),
            .rst                                                    (rst),

            .cin0                                                   (c[0]),
            .in0                                                    (i0_p),
            .in1                                                    (i1_p),
            .out0                                                   (s2),
            .cout0                                                  (c[1]),
            .m_i                                                    ({m_o_p[0], s0}),
            .m_o                                                    ({m_o_p[1], s1})
        );

    end
endgenerate

endmodule




































module shift_adder_6 #(
    W=384,
    S0=0,
    S1=1,
    S2=2,
    S3=3,
    S4=4,
    S5=5,
    C=0,
    M=1,
    R=1,
    R0=0,
    R1=1
)(
    input wire                                                      clk,
    input wire                                                      rst,

    // input wire [1-1:0]                                              cin0,
    input wire [W-1:0]                                              in0,
    input wire [W-1:0]                                              in1,
    input wire [W-1:0]                                              in2,
    input wire [W-1:0]                                              in3,
    input wire [W-1:0]                                              in4,
    input wire [W-1:0]                                              in5,
    // input wire [M-1:0]                                              m_i,
    // output logic [M-1:0]                                            m_o,
    output logic [W-1:0]                                            out0
    // output logic [1-1:0]                                            cout0
);

wire [W-1:0] i0, i1, i2, i3, i4, i5;

wire [W-1:0] c01_s;
wire [W-1:0] c01_c0;
wire [W-1:0] c01_c1;
wire [W-1:0] c02_s;
wire [W-1:0] c02_c;

assign i0 = in0 << S0;
assign i1 = in1 << S1;
assign i2 = in2 << S2;
assign i3 = in3 << S3;
assign i4 = in4 << S4;
assign i5 = in5 << S5;

red_6_3 #(
    .W(W),
    .R(R0)
) red_6_3_inst (
    .in0(i0),
    .in1(i1),
    .in2(i2),
    .in3(i3),
    .in4(i4),
    .in5(i5),
    .sout(c01_s),
    .cout0(c01_c0),
    .cout1(c01_c1),
    .clk(clk)
);

red_3_2 #(
    .W(W),
    .R(R1)
) red_3_2_inst (
    .i0(c01_s),
    .i1(c01_c0 << 1),
    .i2(c01_c1 << 2),
    .s(c02_s),
    .c(c02_c),
    .clk(clk)
);

piped_adder #(
    .W(W),
    .R(R),
    .C(C)
) a0_inst (
    .clk(clk),
    .rst(rst),
    .cin0('0),
    .in0(c02_s),
    .in1(c02_c << 1),
    .out0(out0),
    .cout0(),
    .m_i(),
    .m_o()
);

endmodule
































module red_6_3 #(
    W = 1,
    R = 0
)(
    input wire [W-1:0] in0,
    input wire [W-1:0] in1,
    input wire [W-1:0] in2,
    input wire [W-1:0] in3,
    input wire [W-1:0] in4,
    input wire [W-1:0] in5,
    output logic [W-1:0][1-1:0] sout,
    output logic [W-1:0][1-1:0] cout0,
    output logic [W-1:0][1-1:0] cout1,
    input wire [1-1:0] clk
);

generate
    for (genvar g_i = 0; g_i < W; g_i ++) begin
        logic [3-1:0] ss;

        assign ss = in0[g_i]+in1[g_i]+in2[g_i]+in3[g_i]+in4[g_i]+in5[g_i];

        if (R == 0) begin
            assign sout[g_i] = ss[0];
            assign cout0[g_i] = ss[1];
            assign cout1[g_i] = ss[2];
        end else begin
            always_ff@(posedge clk) sout[g_i] <= ss[0];
            always_ff@(posedge clk) cout0[g_i] <= ss[1];
            always_ff@(posedge clk) cout1[g_i] <= ss[2];
        end

    end
endgenerate


endmodule










module red_3_2 #(
    W = 1,
    R = 1
)(
    input wire [W-1:0] i0,
    input wire [W-1:0] i1,
    input wire [W-1:0] i2,
    output logic [W-1:0] s,
    output logic [W-1:0] c,
    input wire [1-1:0] clk

);

generate
    for (genvar g_i = 0; g_i < W; g_i ++) begin
        logic [2-1:0] ss;

        assign ss = i0[g_i] + i1[g_i] + i2[g_i];

        if (R == 0) begin
            assign s[g_i] = ss[0];
            assign c[g_i] = ss[1];
        end else begin
            always_ff@(posedge clk) s[g_i] <= ss[0];
            always_ff@(posedge clk) c[g_i] <= ss[1];
        end

    end
endgenerate

endmodule

`default_nettype wire
