// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

// Gray code to binary decoder

module gray_to_bin #(parameter int WIDTH = 2)
  (
   input  logic [WIDTH-1:0] i,
   output logic [WIDTH-1:0] o
   );

  always_comb begin
    for (int w = 0; w < WIDTH; w++) begin
      o[w] = ^(i >> w);
    end
  end

endmodule
