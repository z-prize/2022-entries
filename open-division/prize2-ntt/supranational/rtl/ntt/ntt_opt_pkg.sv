// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

package ntt_opt_pkg;

  // Optimize by storing only those twiddles which are unique and replicating
  // as needed.  With Gentleman-Sande, each butterfly level eliminates the need 
  // for 1/2 of twiddles.
  localparam OPTIMIZE_TWIDDLE_PIPELINING = 1;


  // In a 2 pass NTT as we use for 2^18 or 2^24 the first pass (pass==0) has
  // unique twiddles at the input for each point.  For the second pass (pass==1)
  // the twiddles for each lane are identical.  We can use this to save on
  // RAMs.
  localparam OPTIMIZE_PASS1_TWIDDLES = 1;

  // The final butterfly stage uses a twiddle factor of W0 = 1, and the second
  // to last stage uses twiddle factors of W0 or W2 = (1 << 48) for the choice
  // of generator = 7.  As such, we can optimize these stages compared to the
  // more generic stages which need a true 64x64 multiply.
  localparam OPTIMIZE_FINAL_BFLY_STAGES = 1;

endpackage
