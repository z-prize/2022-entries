`default_nettype none

`define Q 'h1AE3A4617C510EAC63B05C06CA1493B1A22D9F300F5138F1EF3622FBA094800170B5D44300000008508C00000000001
// `define K_378 'h0c5e2f3c8bd4a6f76eaac1eeb5440814482f8d517faf2532624e377854a1bbf5646e689d0c1f552322c8dfd08894e0e
`define K_384 'h1784602fbff628a8bf6c1dd0d95a93e5597097dc5f2bb260f4d7cf27ef38fa5eae0237580faa8faf24b7e8444a706c6

module twisted_edwards_prek_full #(
    S=0,
    W=378,
    // WQ=378,WR=384,WM=384,L=4,T0=32'h0000_0999,T1=32'h0000_005F,T2=32'h0000_0999,D_M=41,
    // WQ=377,WR=384,WM=384,L=4,T0=32'h0000_0999,T1=32'h0000_005F,T2=32'h0000_004F,D_M=28,
    // WQ=377,WR=384,WM=384,L=3,T0=32'h07F7_F999,T1=32'h0000_005F,T2=32'h07F7_F999,D_M=35,
    WQ=377,WR=384,WM=384,L=3,T0=32'h07F7_FBBB,T1=32'h0000_005F,T2=32'h07F7_FBBB,D_M=39,
    M=32
)(
    input wire                                                      clk,
    input wire                                                      rst,

    input wire [1-1:0]                                              asub,

    input wire [WQ-1:0]                                             in0_x,
    input wire [WQ-1:0]                                             in0_y,
    input wire [WQ-1:0]                                             in0_z,
    input wire [WQ-1:0]                                             in0_t,
    input wire [WQ-1:0]                                             in1_x,
    input wire [WQ-1:0]                                             in1_y,
    input wire [WQ-1:0]                                             in1_z,
    input wire [WQ-1:0]                                             in1_t,
    input wire [M-1:0]                                              m_i,
    output logic [M-1:0]                                            m_o,
    output logic [WQ-1:0]                                           out0_x,
    output logic [WQ-1:0]                                           out0_y,
    output logic [WQ-1:0]                                           out0_z,
    output logic [WQ-1:0]                                           out0_t
);

generate
    if (S == 0) begin

logic [2-1:0] st;

logic [2-1:0]                                           m_o1;

logic [M-1:0]                                           m_o0_r;
logic [M-1:0]                                           m_o1_r;
logic [WQ-1:0]                                          out0_x_r;
logic [WQ-1:0]                                          out0_y_r;
logic [WQ-1:0]                                          out0_z_r;
logic [WQ-1:0]                                          out0_t_r;
logic [WQ-1:0]                                          out1_r;

logic [WQ-1:0]                                          out1;

logic [M-1:0]                                           te_m_i;
logic [4-1:0][WQ-1:0]                                   te_in0;
logic [3-1:0][WQ-1:0]                                   te_in1;

logic [WQ-1:0]                                          in0_z_r;
logic [WQ-1:0]                                          in1_z_r;
logic [WQ-1:0]                                          in1_t_r;

always_ff@(posedge clk) begin
    case (st)
        0: begin
            // if valid and not full-pipe
            te_m_i <= m_i;
            te_m_i[0] <= m_i[0] & ~m_i[1];
            te_m_i[1] <= m_i[0] & m_i[1];

            in0_z_r <= in0_z;
            in1_z_r <= in1_z;
            in1_t_r <= in1_t;

            te_in0[0] <= in0_x;
            te_in0[1] <= in0_y;
            te_in0[2] <= in0_z;
            te_in0[3] <= in0_t;

            te_in1[0] <= (asub & |in1_x) ? `Q - in1_x : in1_x;
            te_in1[1] <= in1_y;
            te_in1[2] <= m_i[1] ? `K_384 : (asub & |in1_t) ? `Q - in1_t : in1_t;

            // if valid and full-pipe
            if (m_i[0] & m_i[1])
                st <= 1;
        end
        1: begin
            te_in0[3] <= in0_z_r;
            te_in1[2] <= in1_z_r;

            st <= 2;
        end
        2: begin
            te_m_i[0] <= 0;
            te_m_i[1] <= 0;

            te_in0[3] <= out1; // t0*K
            te_in1[2] <= in1_t_r;

            if (m_o1[1])
                st <= 3;
        end
        3: begin

            te_m_i[0] <= m_o1[1];
            te_m_i[1] <= 1'b0;

            te_in0[2] <= out1; // z0*z1

            if (m_o1[1])
                st <= 0;
        end
    endcase
    if (rst)
        st <= 0;
end

always_ff@(posedge clk) out0_x              <= out0_x_r;
always_ff@(posedge clk) out0_y              <= out0_y_r;
always_ff@(posedge clk) out0_z              <= out0_z_r;
always_ff@(posedge clk) out0_t              <= out0_t_r;
always_ff@(posedge clk) out1                <= out1_r;
always_ff@(posedge clk) m_o                 <= m_o0_r;
always_ff@(posedge clk) m_o1                <= m_o1_r;

twisted_edwards_prek #(
    .M(M),
    .W(W),.WQ(WQ),.WR(WR),.WM(WM),.L(L),.T0(T0),.T1(T1),.T2(T2),.D_M(D_M)
) twisted_edwards_prek_inst (
    .clk(clk),
    .rst(rst),

    .in0_x(te_in0[0]),
    .in0_y(te_in0[1]),
    .in0_z(te_in0[2]),
    .in0_t(te_in0[3]),

    .in1_x(te_in1[0]),
    .in1_y(te_in1[1]),
    .in1_z('0),
    .in1_t(te_in1[2]),

    .out0_x(out0_x_r),
    .out0_y(out0_y_r),
    .out0_z(out0_z_r),
    .out0_t(out0_t_r),
    .out1(out1_r),

    .m_i(te_m_i),
    .m_o0(m_o0_r),
    .m_o1(m_o1_r)
);



























// always_ff@(posedge clk)
// if (mont_i_v)
// $display("%t: %m.mont_i: %x %x",$time
//     , mont_in0[0+:32]
//     , mont_in1[0+:32]
// );
// always_ff@(posedge clk)
// if (mont_o_v)
// $display("%t: %m.mont_o: %x",$time
//     , mont_out0[0+:32]
// );

// always_ff@(posedge clk)
// if (te_m_i[0] & te_m_i[1])
// $display("%t: %m.TE_full: %x %x %x %x - %x %x %x",$time
//     , te_in0[0][0+:32]
//     , te_in0[1][0+:32]
//     , te_in0[2][0+:32]
//     , te_in0[3][0+:32]

//     , te_in1[0][0+:32]
//     , te_in1[1][0+:32]
//     , te_in1[2][0+:32]
// );


    end else begin

        localparam D = 50;

        logic [D-1:0][M-1:0] m_o_p;
        logic [D-1:0][WQ-1:0] p_x;
        logic [D-1:0][WQ-1:0] p_y;
        logic [D-1:0][WQ-1:0] p_z;
        logic [D-1:0][WQ-1:0] p_t;


        always_ff@(posedge clk) begin
            integer i;

            m_o_p[0] <= m_i;
            p_x[0] <= in0_x + in1_x + asub;;//asub ? in0_x - in1_x : in0_x + in1_x;
            p_y[0] <= in0_y + in1_y + asub;;//asub ? in0_y - in1_y : in0_y + in1_y;
            p_z[0] <= in0_z + in1_z + asub;;//asub ? in0_z - in1_z : in0_z + in1_z;
            p_t[0] <= in0_t + in1_t + asub;;//asub ? in0_t - in1_t : in0_t + in1_t;
            for (i = 1; i < D; i ++) begin
                m_o_p[i] <= m_o_p[i-1];
                p_x[i] <= p_x[i-1];
                p_y[i] <= p_y[i-1];
                p_z[i] <= p_z[i-1];
                p_t[i] <= p_t[i-1];
            end
        end

        assign m_o = m_o_p[D-1];
        assign out0_x = p_x[D-1];
        assign out0_y = p_y[D-1];
        assign out0_z = p_z[D-1];
        assign out0_t = p_t[D-1];

    end
endgenerate


endmodule

`default_nettype wire
