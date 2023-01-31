// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

module ram_1w1r_1clk
  #(
    parameter ADDR_WIDTH  = 8,
    parameter WORDS       = (1<<ADDR_WIDTH),
    parameter DATA_WIDTH  = 32,
    parameter PIPE_STAGES = 2
    )
  (
   input                          clk_i,

   input logic [ADDR_WIDTH-1:0]   a_a_i,
   input logic [DATA_WIDTH-1:0]   a_wd_i,
   input logic                    a_we_i,
   
   input logic [ADDR_WIDTH-1:0]   b_a_i,
   input logic                    b_re_i,
   output logic [DATA_WIDTH-1: 0] b_rd_o
   );

`ifndef SYNTHESIS
  initial begin
    $display("BOM %m %d X %d",WORDS,DATA_WIDTH);
  end
`endif

// xpm_memory_sdpram: Simple Dual Port RAM
// Xilinx Parameterized Macro, version 2021.2
xpm_memory_sdpram #(
 .ADDR_WIDTH_A(ADDR_WIDTH), // DECIMAL
 .ADDR_WIDTH_B(ADDR_WIDTH), // DECIMAL
 .AUTO_SLEEP_TIME(0), // DECIMAL
 .BYTE_WRITE_WIDTH_A(DATA_WIDTH), // DECIMAL
 .CASCADE_HEIGHT(0), // DECIMAL
 .CLOCKING_MODE("common_clock"), // String
 .ECC_MODE("no_ecc"), // String
 .MEMORY_INIT_FILE("none"), // String
 .MEMORY_INIT_PARAM("0"), // String
 .MEMORY_OPTIMIZATION("true"), // String
 .MEMORY_PRIMITIVE("auto"), // String
 .MEMORY_SIZE(DATA_WIDTH<<ADDR_WIDTH), // DECIMAL
 .MESSAGE_CONTROL(0), // DECIMAL
 .READ_DATA_WIDTH_B(DATA_WIDTH), // DECIMAL
 .READ_LATENCY_B(PIPE_STAGES), // DECIMAL
 .READ_RESET_VALUE_B("0"), // String
 .RST_MODE_A("SYNC"), // String
 .RST_MODE_B("SYNC"), // String
 .SIM_ASSERT_CHK(0), // DECIMAL; 0=disable simulation messages, 1=enable simulation messages
 .USE_EMBEDDED_CONSTRAINT(0), // DECIMAL
 .USE_MEM_INIT(1), // DECIMAL
 .USE_MEM_INIT_MMI(0), // DECIMAL
 .WAKEUP_TIME("disable_sleep"), // String
 .WRITE_DATA_WIDTH_A(DATA_WIDTH), // DECIMAL
 .WRITE_MODE_B("no_change"), // String
 .WRITE_PROTECT(1) // DECIMAL
)
xpm_memory_tdpram_inst (
 .dbiterrb(), // 1-bit output: Status signal to indicate double bit error occurrence
 // on the data output of port A.
 .doutb(b_rd_o), // READ_DATA_WIDTH_B-bit output: Data output for port B read operations.
 .sbiterrb(), // 1-bit output: Status signal to indicate single bit error occurrence
 // on the data output of port B.
 .addra(a_a_i), // ADDR_WIDTH_A-bit input: Address for port A write and read operations.
 .addrb(b_a_i), // ADDR_WIDTH_B-bit input: Address for port B write and read operations.
 .clka(clk_i), // 1-bit input: Clock signal for port A. Also clocks port B when
 // parameter CLOCKING_MODE is "common_clock".
 .clkb(clk_i), // 1-bit input: Clock signal for port B when parameter CLOCKING_MODE is
 // "independent_clock". Unused when parameter CLOCKING_MODE is
 // "common_clock".
 .dina(a_wd_i), // WRITE_DATA_WIDTH_A-bit input: Data input for port A write operations.
 .ena(a_we_i), // 1-bit input: Memory enable signal for port A. Must be high on clock
 // cycles when read or write operations are initiated. Pipelined
 // internally.
 .enb(b_re_i), // 1-bit input: Memory enable signal for port B. Must be high on clock
 // cycles when read or write operations are initiated. Pipelined
 // internally.
 .injectdbiterra(1'b0), // 1-bit input: Controls double bit error injection on input data when
 // ECC enabled (Error injection capability is not available in
 // "decode_only" mode).
 .injectsbiterra(1'b0), // 1-bit input: Controls single bit error injection on input data when
 // ECC enabled (Error injection capability is not available in
 // "decode_only" mode).
 .regceb(b_re_i), // 1-bit input: Clock Enable for the last register stage on the output
 // data path.
 .rstb(1'b0), // 1-bit input: Reset signal for the final port B output register stage.
 // Synchronously resets output port doutb to the value specified by
 // parameter READ_RESET_VALUE_B.
 .sleep(1'b0), // 1-bit input: sleep signal to enable the dynamic power saving feature.
 .wea(a_we_i) // WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-bit input: Write enable vector
 // for port A input data port dina. 1 bit wide when word-wide writes are
 // used. In byte-wide write configurations, each bit controls the
 // writing one byte of dina to address addra. For example, to
 // synchronously write only bits [15-8] of dina when WRITE_DATA_WIDTH_A
 // is 32, wea would be 4'b0010.
);
// End of xpm_memory_tdpram_inst instantiation
endmodule
