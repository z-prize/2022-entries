/*
 * Copyright (C) 2022 DZK
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _NTT_FUNC_H_
#define _NTT_FUNC_H_

#include "arithmetic.hpp"
#include "hls_stream.h"

//void NTT_2_3_in_place(GF (&in)[8]);
//void NTT_2_12_in_place(GF (&in)[4096]);
//void NTT_2_18_in_place(GF (&in)[262144]);
void NTT_2_24_in_place(GF (&in)[16777216], GF powers[16777216], GF (&out)[16777216]);

#endif
