`default_nettype none










`define SHADD_6(_W, _O, _S0, _S1, _S2, _S3, _S4, _S5, _I0, _I1, _I2, _I3, _I4, _I5) shift_adder_6 #(.W(_W),.S0(_S0),.S1(_S1),.S2(_S2),.S3(_S3),.S4(_S4),.S5(_S5),.R(1)) ``_O``_shadd6_inst (.in0(_I0),.in1(_I1),.in2(_I2),.in3(_I3),.in4(_I4),.in5(_I5),.out0(_O),.clk(clk),.rst(rst));



































module mul_wide #(
    W=384,W0=W,W1=W,L=3,T=32'h07F7_F999,
    W2=W/2,
    R_I=0,
    CT = T[0 +: 4],
    ST = T >> 4,
    M=32,
    S=0
)(
    input wire                                                      clk,
    input wire                                                      rst,

    input wire [W0-1:0]                                             in0,
    input wire [W1-1:0]                                             in1,
    input wire [M-1:0]                                              m_i,
    output logic [M-1:0]                                            m_o,
    output logic [W0+W1-1:0]                                        out0
);

logic [W0-1:0] in0_r;
logic [W1-1:0] in1_r;
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


    //      999999999     
    //    99:::::::::99   
    //  99:::::::::::::99 
    // 9::::::99999::::::9
    // 9:::::9     9:::::9
    // 9:::::9     9:::::9
    //  9:::::99999::::::9
    //   99::::::::::::::9
    //     99999::::::::9 
    //          9::::::9  
    //         9::::::9   
    //        9::::::9    
    //       9::::::9     
    //      9::::::9      
    //     9::::::9       
    //    99999999        
    // Karatsuba-2
    if (CT == 9) begin: T9

        localparam WN = W2 + (1 << (L-1));

        logic [2-1:0][W2-1:0] x;
        logic [2-1:0][W2-1:0] y;
        logic [2-1:0][W2-1:0] x_p;
        logic [2-1:0][W2-1:0] y_p;
        logic [W-1:0] z0, z2;
        // logic [W-1:0] z0_p, z2_p;
        logic [WN*2-1:0] z1;
        // logic [WN*2-1:0] z1_p;
        logic [W-1:0] m0, m2;
        logic [WN*2-1:0] m1;
        logic [WN-1:0] a;
        logic [WN-1:0] b;
        logic [3-1:0][M-1:0] m_o_p;

        assign x = in0;
        assign y = in1;

        always_ff@(posedge clk) z0 <= m0;
        always_ff@(posedge clk) z1 <= m1 - m2 - m0;
        always_ff@(posedge clk) z2 <= m2;
        always_ff@(posedge clk) m_o_p[2] <= m_o_p[1];

        always_ff@(posedge clk) out0 <= {z2, {W{1'b0}}} + {z1, {W2{1'b0}}} + z0;
        always_ff@(posedge clk) m_o <= m_o_p[2];

        if (WN-1>=W2+1)
            assign a[WN-1:W2+1] = '0;
        if (WN-1>=W2+1)
            assign b[WN-1:W2+1] = '0;
        piped_adder #(
            .W(W2),
            .R(1),
            .C(W2 >= 1280 ? 1 : 0)
        ) sx_inst (
            .clk(clk),
            .rst(rst),
            .cin0('0),
            .in0(x[0]),
            .in1(x[1]),
            .out0(a[0+:W2]),
            .cout0(a[W2]),
            .m_i(),
            .m_o()
        );
        piped_adder #(
            .W(W2),
            .R(1),
            .C(W2 >= 1280 ? 1 : 0),
            .M(M+W+W)
        ) sy_inst (
            .clk(clk),
            .rst(rst),
            .cin0('0),
            .in0(y[0]),
            .in1(y[1]),
            .out0(b[0+:W2]),
            .cout0(b[W2]),
            .m_i({m_i, x, y}),
            .m_o({m_o_p[0], x_p, y_p})
        );

        mul_wide #(
            .L(L-1),
            .W(W2),
            .T(ST)
        ) m0_inst(
            .clk(clk),
            .rst(rst),
            .in0(x_p[0]),
            .in1(y_p[0]),
            .out0(m0)
        );
        mul_wide #(
            .L(L-1),
            .W(WN),
            .T(ST),
            .M(M)
        ) m1_inst(
            .clk(clk),
            .rst(rst),
            .in0(a),
            .in1(b),
            .out0(m1),
            .m_i(m_o_p[0]),
            .m_o(m_o_p[1])
        );
        mul_wide #(
            .L(L-1),
            .W(W2),
            .T(ST),
            .M(M)
        ) m2_inst(
            .clk(clk),
            .rst(rst),
            .in0(x_p[1]),
            .in1(y_p[1]),
            .out0(m2),
            .m_i(),
            .m_o()
        );

    end else if (CT == 15 && ST[0+:4] == 5) begin: T15_5
        if (W == 384) begin
            `include "mul_const_BLS12_377_QP_NAF_384.svh"
        end
    end else if (CT == 15 && ST[0+:4] == 7) begin: T15_7

        localparam DEPTH = W0 < 27 ? 1 : W0 <= 26*2 ? 2 : 1+4;

        if (W0 < 27) begin
        `include "mul_wide_17nx26_dsp48e2.svh"
        end else if (W0 <= 26*2) begin

            logic [26+W1-1:0] m_0;
            logic [W0-26+W1-1:0] m_1;

            mul_wide #(.W0(26)   , .W1(W1), .T(T)) m_0_inst (.clk(clk), .rst(rst), .in0(in0[  0 +: 26])   , .in1(in1), .out0(m_0));
            mul_wide #(.W0(W0-26), .W1(W1), .T(T)) m_1_inst (.clk(clk), .rst(rst), .in0(in0[ 26 +: W0-26]), .in1(in1), .out0(m_1));

            always_ff@(posedge clk) begin
                out0[0+:26] = m_0[0+:26];
                out0[26+:W0+W1-26] <= m_1 + m_0[26+:W1];
                // out0 <= m_0 + (m_1 << 26);
            end

        end

        logic [DEPTH-1:0][M-1:0] m_o_p;

        always_ff@(posedge clk) begin
            integer i;
            m_o_p[0] <= m_i;
            for (i = 1; i < DEPTH; i ++)
                m_o_p[i] <= m_o_p[i-1];
            end
        assign m_o = m_o_p[DEPTH-1];
    end
endgenerate

endmodule


`default_nettype wire
