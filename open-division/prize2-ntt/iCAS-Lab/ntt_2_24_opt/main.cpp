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
/*******************************************************************************
 * Libs
 */
#include <ap_int.h>

#include <bitset>
#include <fstream>
#include <iostream>
#include <string>

#include "ntt.hpp"

using std::cout;
using std::endl;
/*******************************************************************************
 * Decls
 */
// Paths for data
const std::string IN_FILE_PATH = "/home/<YOURUSERNAME>/ntt_2_24_opt/data/in/";
const std::string OUT_FILE_PATH = "/home/<YOURUSERNAME>/ntt_2_24_opt/data/out/";
const std::string POWERS_FILE_PATH = "/home/<YOURUSERNAME>/ntt_2_24_opt/data/";
/*******************************************************************************
 * File IO Functions
 */
template <typename T, size_t N>
void readin(const int degree, T (&in)[N], const std::string prefix) {
  const uint32_t powered = pow(2, degree);
  std::string fname = IN_FILE_PATH;
  std::ifstream in_file(fname + prefix + std::to_string(degree));
  std::string line;
  uint32_t count = 0;
  if (in_file.is_open()) {
    cout << "Reading File: " << fname << prefix << std::to_string(degree)
         << endl;
    while ((count < powered) && std::getline(in_file, line)) {
      in[count] = GF(std::stoul(line, nullptr, 16));
      ++count;
    }
  }
  in_file.close();
}

template <typename T, size_t N>
void readout(const int degree, T (&out)[N], const std::string prefix) {
  const uint32_t powered = pow(2, degree);
  std::string fname = OUT_FILE_PATH;
  std::ifstream out_file(fname + prefix + std::to_string(degree));
  std::string line;
  uint32_t count = 0;
  if (out_file.is_open()) {
    cout << "Reading File: " << fname << prefix << std::to_string(degree)
         << endl;
    while ((count < powered) && std::getline(out_file, line)) {
      out[count] = GF(std::stoul(line, nullptr, 16));
      ++count;
    }
  }
  out_file.close();
}

template <typename T, size_t N>
void readpowers(const int degree, T (&out)[N], const std::string powers_file) {
  const uint32_t powered = pow(2, degree);
  std::string fname = POWERS_FILE_PATH;
  std::ifstream out_file(fname + powers_file);
  std::string line;
  uint32_t count = 0;
  if (out_file.is_open()) {
    cout << "Reading File: " << fname << powers_file << endl;
    while ((count < powered) && std::getline(out_file, line)) {
      out[count] = GF(std::stoul(line, nullptr, 16));
      ++count;
    }
  }
  out_file.close();
}
/*******************************************************************************
 * Main Function
 */
int main() {
  cout << "LOG --> In main()..." << endl;
  // Prefixes for data files
  const std::string RANDOM_PREFIX = "fully_random_2_";
  const std::string POWERS_FILE = "omega_powers";

  // Create arrays for storing values of input/output
  cout << "LOG --> Init arrays..." << endl;
  static ap_uint<65> fully_random_in[16777216] = {};
  static ap_uint<65> out[16777216] = {};
  static GF omega_powers[16777216] = {};
  static GF res[16777216] = {};

  // Read the data files
  cout << "LOG --> Reading data from files..." << endl;
  readin(24, fully_random_in, RANDOM_PREFIX);
  // readout(24, fully_random_out, RANDOM_PREFIX);

  // Read the precomputed Omega powers data file
  cout << "LOG --> Reading Omega powers from files..." << endl;
  readpowers(24, omega_powers, POWERS_FILE);

  // Sanity Check for ensuring data was read from files
  cout << "Initial input:" << endl;
  for (int i = 0; i < 8; ++i) {
    cout << fully_random_in[i].to_string(16) << endl;
  }
  cout << "Powers of Omega:" << endl;
  for (int i = 0; i < 8; ++i) {
    cout << omega_powers[i].to_string(16) << endl;
  }

  // Do the NTT algorithm
  // static hls::stream<GF> out_stream;
  cout << "LOG --> Starting NTT computation..." << endl;
  NTT_2_24_in_place(fully_random_in, omega_powers, out);

  // Take in stream data in bit-reverse order
  cout << "LOG --> Post-computation bit-reversal reordering..." << endl;
  typedef ap_uint<24> INDEX;
  INDEX i = INDEX(0);
  do {
    INDEX ri = i;
    ri.reverse();
    res[ri] = out[i];
    i++;
  } while (!i.iszero());

  // Write results to file
  cout << "LOG --> Writing in-order results to file..." << endl;
  std::ofstream outfile;
  outfile.open("result_2_24.out");
  for (uint32_t i = 0; i < 16777216; ++i) {
    outfile << res[i].to_string(16) << endl;
  }
  outfile.close();

  /*// Check the results vs. the expected output
  cout << "LOG --> Checking results against expected output..." << endl;
  for (uint32_t i = 0; i < 16777216; ++i) {
          // If the result is not the same as the expected.
          if (res[i] != fully_random_out[i]) {
                  cout << "Error: NTT result [" << i << "] should be "
                           << fully_random_out[i].to_string(16) << ", but the
  output is "
                           << res[i].to_string(16) << endl;
                  return 1;
          }
  }
  cout << "All test passed." << endl;*/
  return 0;
}
