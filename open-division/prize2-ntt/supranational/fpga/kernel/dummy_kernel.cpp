// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

//-----------------------------------------------------------------------------
// kernel: nantucket
//
// Purpose: Since the HLS flow is not used the sw_emu make target has little
//          value.  This code implements a dummy kernel which allows the
//          sw_emu flow to build but it doesn't compute any results.
//-----------------------------------------------------------------------------
#include <string.h>
#include "hls_half.h"
#include "ap_axi_sdata.h"
#include "hls_stream.h"

// Function declaration/Interface pragmas to match RTL Kernel
extern "C" void nantucket (
    unsigned char chicken_bits,
    int* axi00_ptr0,
    int* axi01_ptr0,
    int* axi02_ptr0,
    int* axi03_ptr0,
    int* axi04_ptr0,
    int* axi05_ptr0,
    int* axi06_ptr0,
    int* axi07_ptr0,
    int* axi08_ptr0,
    int* axi09_ptr0,
    int* axi10_ptr0,
    int* axi11_ptr0,
    int* axi12_ptr0,
    int* axi13_ptr0,
    int* axi14_ptr0,
    int* axi15_ptr0,
    int* axi16_ptr0,
    int* axi17_ptr0,
    int* axi18_ptr0,
    int* axi19_ptr0,
    int* axi20_ptr0,
    int* axi21_ptr0,
    int* axi22_ptr0,
    int* axi23_ptr0,
    int* axi24_ptr0,
    int* axi25_ptr0,
    int* axi26_ptr0,
    int* axi27_ptr0,
    int* axi28_ptr0,
    int* axi29_ptr0,
    int* axi30_ptr0,
    int* axi31_ptr0
) {
    #pragma HLS INTERFACE m_axi port=axi00_ptr0 offset=slave bundle=m00_axi
    #pragma HLS INTERFACE m_axi port=axi01_ptr0 offset=slave bundle=m01_axi
    #pragma HLS INTERFACE m_axi port=axi02_ptr0 offset=slave bundle=m02_axi
    #pragma HLS INTERFACE m_axi port=axi03_ptr0 offset=slave bundle=m03_axi
    #pragma HLS INTERFACE m_axi port=axi04_ptr0 offset=slave bundle=m04_axi
    #pragma HLS INTERFACE m_axi port=axi05_ptr0 offset=slave bundle=m05_axi
    #pragma HLS INTERFACE m_axi port=axi06_ptr0 offset=slave bundle=m06_axi
    #pragma HLS INTERFACE m_axi port=axi07_ptr0 offset=slave bundle=m07_axi
    #pragma HLS INTERFACE m_axi port=axi08_ptr0 offset=slave bundle=m08_axi
    #pragma HLS INTERFACE m_axi port=axi09_ptr0 offset=slave bundle=m09_axi
    #pragma HLS INTERFACE m_axi port=axi10_ptr0 offset=slave bundle=m10_axi
    #pragma HLS INTERFACE m_axi port=axi11_ptr0 offset=slave bundle=m11_axi
    #pragma HLS INTERFACE m_axi port=axi12_ptr0 offset=slave bundle=m12_axi
    #pragma HLS INTERFACE m_axi port=axi13_ptr0 offset=slave bundle=m13_axi
    #pragma HLS INTERFACE m_axi port=axi14_ptr0 offset=slave bundle=m14_axi
    #pragma HLS INTERFACE m_axi port=axi15_ptr0 offset=slave bundle=m15_axi
    #pragma HLS INTERFACE m_axi port=axi16_ptr0 offset=slave bundle=m16_axi
    #pragma HLS INTERFACE m_axi port=axi17_ptr0 offset=slave bundle=m17_axi
    #pragma HLS INTERFACE m_axi port=axi18_ptr0 offset=slave bundle=m18_axi
    #pragma HLS INTERFACE m_axi port=axi19_ptr0 offset=slave bundle=m19_axi
    #pragma HLS INTERFACE m_axi port=axi20_ptr0 offset=slave bundle=m20_axi
    #pragma HLS INTERFACE m_axi port=axi21_ptr0 offset=slave bundle=m21_axi
    #pragma HLS INTERFACE m_axi port=axi22_ptr0 offset=slave bundle=m22_axi
    #pragma HLS INTERFACE m_axi port=axi23_ptr0 offset=slave bundle=m23_axi
    #pragma HLS INTERFACE m_axi port=axi24_ptr0 offset=slave bundle=m24_axi
    #pragma HLS INTERFACE m_axi port=axi25_ptr0 offset=slave bundle=m25_axi
    #pragma HLS INTERFACE m_axi port=axi26_ptr0 offset=slave bundle=m26_axi
    #pragma HLS INTERFACE m_axi port=axi27_ptr0 offset=slave bundle=m27_axi
    #pragma HLS INTERFACE m_axi port=axi28_ptr0 offset=slave bundle=m28_axi
    #pragma HLS INTERFACE m_axi port=axi29_ptr0 offset=slave bundle=m29_axi
    #pragma HLS INTERFACE m_axi port=axi30_ptr0 offset=slave bundle=m30_axi
    #pragma HLS INTERFACE m_axi port=axi31_ptr0 offset=slave bundle=m31_axi
    #pragma HLS INTERFACE s_axilite port=chicken_bits bundle=control
    #pragma HLS INTERFACE s_axilite port=axi00_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi01_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi02_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi03_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi04_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi05_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi06_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi07_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi08_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi09_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi10_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi11_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi12_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi13_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi14_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi15_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi16_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi17_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi18_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi19_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi20_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi21_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi22_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi23_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi24_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi25_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi26_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi27_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi28_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi29_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi30_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=axi31_ptr0 bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE ap_ctrl_hs port=return

    unsigned long *base [32];

    base[0]  = (unsigned long *) axi00_ptr0;
    base[1]  = (unsigned long *) axi01_ptr0;
    base[2]  = (unsigned long *) axi02_ptr0;
    base[3]  = (unsigned long *) axi03_ptr0;
    base[4]  = (unsigned long *) axi04_ptr0;
    base[5]  = (unsigned long *) axi05_ptr0;
    base[6]  = (unsigned long *) axi06_ptr0;
    base[7]  = (unsigned long *) axi07_ptr0;
    base[8]  = (unsigned long *) axi08_ptr0;
    base[9]  = (unsigned long *) axi09_ptr0;
    base[10] = (unsigned long *) axi10_ptr0;
    base[11] = (unsigned long *) axi11_ptr0;
    base[12] = (unsigned long *) axi12_ptr0;
    base[13] = (unsigned long *) axi13_ptr0;
    base[14] = (unsigned long *) axi14_ptr0;
    base[15] = (unsigned long *) axi15_ptr0;
    base[16] = (unsigned long *) axi16_ptr0;
    base[17] = (unsigned long *) axi17_ptr0;
    base[18] = (unsigned long *) axi18_ptr0;
    base[19] = (unsigned long *) axi19_ptr0;
    base[20] = (unsigned long *) axi20_ptr0;
    base[21] = (unsigned long *) axi21_ptr0;
    base[22] = (unsigned long *) axi22_ptr0;
    base[23] = (unsigned long *) axi23_ptr0;
    base[24] = (unsigned long *) axi24_ptr0;
    base[25] = (unsigned long *) axi25_ptr0;
    base[26] = (unsigned long *) axi26_ptr0;
    base[27] = (unsigned long *) axi27_ptr0;
    base[28] = (unsigned long *) axi28_ptr0;
    base[29] = (unsigned long *) axi29_ptr0;
    base[30] = (unsigned long *) axi30_ptr0;
    base[31] = (unsigned long *) axi31_ptr0;
}
