`default_nettype none

module shift_reg #(
    W=1,
    D=1
)(
    input wire  [1-1:0]                                             clk,
    input wire  [1-1:0]                                             rst,
    input wire  [W-1:0]                                             in0,
    output logic[W-1:0]                                             out0
);

logic [D-1:0][W-1:0] p;
assign out0 = p[0];
always_ff@(posedge clk) begin
    integer i;
    p[D-1] <= in0;
    for (i = 0; i < D-1; i ++)
        p[i] <= p[i+1];
end
endmodule

module f_add_sub #(
    W=1,
    Q=0,
    ADD=1,
    RED=1
)(
    input wire  [1-1:0]                                             clk,
    input wire  [1-1:0]                                             rst,
    input wire  [W-1:0]                                             in0,
    input wire  [W-1:0]                                             in1,
    output logic[W-1:0]                                             out0
);

logic [W+1-1:0] c02_s;
logic [W+1-1:0] not_q;

assign not_q = -Q;

generate
    if (ADD) begin
        if (RED == 1) begin

            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c00_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(in0),
                .in1(in1),
                .out0(c02_s[0+:W]),
                .cout0(c02_s[W]),
                .m_i(),
                .m_o()
            );

            piped_adder #(
                .W(W+1),
                .R(1),
                .C(1)
            ) c02_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(c02_s[0+:W+1]),
                .in1(c02_s >= Q ? not_q : 0),
                .out0(out0),
                .cout0(),
                .m_i(),
                .m_o()
            );

            // always_ff@(posedge clk) out0 <= s >= Q ? s-Q : s;
        end else begin
            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c00_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(in0),
                .in1(in1),
                .out0(out0),
                .cout0(),
                .m_i(),
                .m_o()
            );
            // always_ff@(posedge clk) out0 <= s;
        end
    end else begin
        if (RED == 1) begin

            logic [1-1:0] c02_underflow;

            piped_adder #(
                .W(W),
                .R(1),
                .C(1),
                .M(1)
            ) c00_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0(1'b1),
                .in0(in0),
                .in1(~in1),
                .out0(c02_s[0+:W]),
                .cout0(),
                .m_i(in0 < in1),
                .m_o(c02_underflow)
            );

            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c02_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0('0),
                .in0(c02_s[0+:W]),
                .in1(c02_underflow ? Q : 0),
                .out0(out0),
                .cout0(),
                .m_i(),
                .m_o()
            );

            // assign s = in0 - in1;
            // always_ff@(posedge clk) out0 <= in0 < in1 ? s+Q : s;
        end else begin
            // piped_adder #(
            //     .W(W),
            //     .R(1),
            //     .C(0)
            // ) c00_0_inst (
            //     .clk(clk),
            //     .rst(rst),
            //     .cin0('0),
            //     .in0(in0),
            //     .in1(Q),
            //     .out0(c02_s),
            //     .cout0()
            // );
            piped_adder #(
                .W(W),
                .R(1),
                .C(1)
            ) c02_0_inst (
                .clk(clk),
                .rst(rst),
                .cin0(1'b1),
                .in0(in0 + Q),
                .in1(~in1),
                .out0(out0),
                .cout0()
            );
            // assign s = (Q + in0) - in1;
            // always_ff@(posedge clk) out0 <= s;
        end
    end
endgenerate

endmodule


`define Q 'h1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
`define K 'h1784602fbff628a8bf6c1dd0d95a93e5597097dc5f2bb260f4d7cf27ef38fa5eae0237580faa8faf24b7e8444a706c6

`define ADD(___, _, __, ____) f_add_sub #(.W(W),.Q(`Q),.ADD(1),.RED(____)) f_add_sub_``___`` (.clk(clk),.rst(rst),.in0(_),.in1(__),.out0(___));
`define SUB(___, _, __, ____) f_add_sub #(.W(W),.Q(`Q),.ADD(0),.RED(____)) f_add_sub_``___`` (.clk(clk),.rst(rst),.in0(_),.in1(__),.out0(___));
`define MUL(___, _, __) mul_montgomery #(.WI(W),.WQ(WQ),.WR(WR),.WM(WM),.L(L),.T0(T0),.T1(T1),.T2(T2)) mul_mont_inst_``___`` (.clk(clk),.rst(rst),.in0(_),.in1(__),.out0(___));
`define PIP(___, _, _W, __) shift_reg #(.W(_W), .D(__)) shift_reg_``___`` (.clk(clk), .rst(rst), .in0(_), .out0(___));

module twisted_edwards_prek #(
    W=378,
    // WQ=378,WR=384,WM=384,L=3,T0=32'h0000_0999,T1=32'h0000_005F,T2=32'h0000_004F,
    // WQ=378,WR=384,WM=384,L=3,T0=32'h0000_0999,T1=32'h0000_005F,T2=32'h0000_004f,D_M=28,
    // WQ=378,WR=384,WM=384,L=3,T0=32'h07F7_F999,T1=32'h0000_005F,T2=32'h0000_004F,D_M=29,
    // WQ=377,WR=384,WM=384,L=3,T0=32'h07F7_F999,T1=32'h0000_005F,T2=32'h07F7_F999,D_M=35,
    WQ=377,WR=384,WM=384,L=3,T0=32'h07F7_FBBB,T1=32'h0000_005F,T2=32'h07F7_FBBB,D_M=39,
    // WQ=378,WR=384,WM=384,L=3,T0=32'h0000_0000,T1=32'h0000_0000,T2=32'h0000_0000,D_M=7,
    // WQ=378,WR=384,WM=384,L=4,T0=32'h0000_0999,T1=32'h0000_0999,T2=32'h0000_0999,D_M=73,
    // WQ=378,WR=384,WM=384,L=4,T0=32'h0000_006F,T1=32'h0000_0999,T2=32'h0000_0999,D_M=58,
    // WQ=377,WR=377,WM=378,L=4,T0=32'h0000_0AAA,T1=32'h0000_0AAA,T2=32'h0000_0AAA,D_M=79,
    // WQ=377,WR=377,WM=378,L=4,T0=32'h0000_AAAA,T1=32'h0000_005F,T2=32'h0000_AAAA,D_M=50,
    // WQ=377,WR=377,WM=378,L=4,T0=32'h0000_AAAA,T1=32'h0000_005F,T2=32'h0000_004F,D_M=40,
    M=32
)(
    input wire                                                      clk,
    input wire                                                      rst,

    input wire [WQ-1:0]                                             in0_x,
    input wire [WQ-1:0]                                             in0_y,
    input wire [WQ-1:0]                                             in0_z,
    input wire [WQ-1:0]                                             in0_t,
    input wire [WQ-1:0]                                             in1_x,
    input wire [WQ-1:0]                                             in1_y,
    input wire [WQ-1:0]                                             in1_z,
    input wire [WQ-1:0]                                             in1_t,
    input wire [M-1:0]                                              m_i,
    output logic [M-1:0]                                            m_o0,
    output logic [M-1:0]                                            m_o1,
    output logic [WQ-1:0]                                           out0_x,
    output logic [WQ-1:0]                                           out0_y,
    output logic [WQ-1:0]                                           out0_z,
    output logic [WQ-1:0]                                           out0_t,
    output logic [WQ-1:0]                                           out1
);

localparam D_A = 2;

generate

logic [W-1:0] in0_x_;
logic [W-1:0] in0_y_;
logic [W-1:0] in0_z_;
logic [W-1:0] in0_t_;
logic [W-1:0] in1_x_;
logic [W-1:0] in1_y_;
logic [W-1:0] in1_z_;
logic [W-1:0] in1_t_;

logic [W-1:0] R1_a;
logic [W-1:0] R2_a;
logic [W-1:0] R3_a;
logic [W-1:0] R4_a;
logic [W-1:0] R5_am;
logic [W-1:0] R6_am;
logic [W-1:0] R7_m;
logic [W-1:0] R8;
logic [W-1:0] R9_m;
logic [W-1:0] R10_aa;
logic [W-1:0] R10_m;
logic [W-1:0] R11_ama;
logic [W-1:0] R12_ma;
logic [W-1:0] R12_ama_x;
logic [W-1:0] R12_ama_z;
logic [W-1:0] R13_ma;
logic [W-1:0] R13_ama_y;
logic [W-1:0] R13_ama_z;
logic [W-1:0] R14_ama;

assign in0_x_ = in0_x;
assign in0_y_ = in0_y;
assign in0_z_ = in0_z;
assign in0_t_ = in0_t;
assign in1_x_ = in1_x;
assign in1_y_ = in1_y;
assign in1_z_ = in1_z;
assign in1_t_ = in1_t;

`SUB(R1_a, in0_y_, in0_x_, 0)
`SUB(R2_a, in1_y_, in1_x_, 0)
`ADD(R3_a, in0_y_, in0_x_, 0)
`ADD(R4_a, in1_y_, in1_x_, 0)

`MUL(R5_am, R1_a, R2_a)
`MUL(R6_am, R3_a, R4_a)
`MUL(R7_m, in0_t_, in1_t_)
assign R8 = in0_z_;
assign R9_m = R7_m;

`ADD(R10_aa, R8, R8, 1)
`SUB(R11_ama, R6_am, R5_am, 0)
`SUB(R12_ma, R10_m, R9_m, 0)
`ADD(R13_ma, R10_m, R9_m, 0)
`ADD(R14_ama, R6_am, R5_am, 0)

`PIP(R10_m, R10_aa, W, D_M-D_A-D_A)
`PIP(R12_ama_x, R12_ma, W, D_A)
`PIP(R12_ama_z, R12_ma, W, D_A)
`PIP(R13_ama_y, R13_ma, W, D_A)
`PIP(R13_ama_z, R13_ma, W, D_A)

`MUL(out0_x, R11_ama, R12_ama_x)
`MUL(out0_y, R13_ama_y, R14_ama)
`MUL(out0_t, R11_ama, R14_ama)
`MUL(out0_z, R12_ama_z, R13_ama_z)

`PIP(m_o0, m_i, M, D_A+D_M+D_A+D_M);
`PIP(m_o1, m_i, M, D_M);

assign out1 = R7_m;


endgenerate

always_ff@(negedge clk)
$display("%t: ----",$time);

endmodule


`default_nettype wire

