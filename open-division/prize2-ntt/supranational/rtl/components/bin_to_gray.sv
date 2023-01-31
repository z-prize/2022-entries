// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Binary to gray code encoder

module bin_to_gray #(parameter int WIDTH = 2)
  (
   input  logic [WIDTH-1:0] i,
   output logic [WIDTH-1:0] o
   );

  assign o = i ^ {1'b0, i[WIDTH-1:1]};

endmodule
