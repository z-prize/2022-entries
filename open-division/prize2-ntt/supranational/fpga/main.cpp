// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
// or the MIT license, see LICENSE-MIT, at your option.
// SPDX-License-Identifier: Apache-2.0 OR MIT

#include "handler_function.h"
/*

                Add your Header Files and definitions


 */

#include "ntt_cfg.h"
#include "util.h"

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>

#if (USE_XRT==1)
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#endif

void wait_for_enter(const std::string &msg) {
    std::cout << msg << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

int main(int argc, char** argv) {

        // ----------------------------------------------------------------------
        // Enhanced template to take filenames from main arguments.

        std::string input_filename  = "./ntt2p18_random_input.txt";
        std::string expect_filename = "./ntt2p18_random_output.txt";
        std::string xclbin_filename = "nantucket.xclbin";

        // If three arguments, assume these are the input and expected points.
        if (argc >= 3) {
          input_filename  = argv[1];
          expect_filename = argv[2];
        }
        // If four arguments, assume the fourth is the xclbin filename.
        if (argc == 4) {
          xclbin_filename = argv[3];
        }

        // ----------------------------------------------------------------------

        std::ofstream outputFile_Handler; // output file handler
        outputFile_Handler.open("ntt_generated_output.txt"); // open file for writing
        std::vector<std::string> input_pts; //pointer to receive the hex data from file reading function
        std::vector<std::string> expect_pts; //pointer to receive the hex data from file reading function
        // set precision for timer e.g pico seconds
        std::cout << std::fixed << std::setprecision(12) << std::left;
        //define kernel time parameter
        std::chrono::duration<double> kernel_time(0);
        // Read data from the file Note: the data will be read from file as string convert it accordingly
        input_pts  = read_data_from_file(input_filename.c_str()); // input test vectors
        expect_pts = read_data_from_file(expect_filename.c_str()); // expected output vectors for comparison
        
        /*

                Convert String data to int/long/unsigned long accordingly and store it in CPU memory and HBM
                The input data should be used as input to the NTT and the expected output data should be
                used for comparison and checking the correctness of the NTT core
         */

        // Make sure the input and expected data sizes are legal.
        assert(input_pts.size() != 0);
        assert(input_pts.size() == expect_pts.size());
        assert(input_pts.size() == (1<<18) || input_pts.size() == (1<<24));

        const unsigned long Nmax = 1<<24;
        const unsigned long Npts = input_pts.size();
        const unsigned int  Nhop = Nmax / Npts;
        const unsigned char chicken_bits = 0;

        std::cout << "N................. " << Npts            << std::endl;
        std::cout << "Input points...... " << input_filename  << std::endl;
        std::cout << "Expected points... " << expect_filename << std::endl;
        std::cout << "xclbin............ " << xclbin_filename << std::endl;

        using namespace ntt_cfg;

#if (USE_XRT==1)

	std::string binaryFile = xclbin_filename;
	int device_index = 0;
	std::cout << "Open the device " << device_index << std::endl;
	auto device = xrt::device(device_index);
	std::cout << "Load the xclbin " << binaryFile << std::endl;
	auto uuid = device.load_xclbin(binaryFile);
	auto krnl = xrt::kernel(device, uuid, "nantucket");

#else

        // Not using HLS so we need to load the FPGA programming file into the device.
        // Also initialize OpenCL environment.
        cl_int err;
        unsigned fileBufSize;
        std::vector<cl::Device> devices = get_xilinx_devices();
        devices.resize(1); // If multiple devices, choose the first.
        cl::Device device = devices[0];
        cl::Context context(device, NULL, NULL, NULL, &err);
        char *fileBuf = read_binary_file(xclbin_filename, fileBufSize);
        cl::Program::Binaries bins{{fileBuf, fileBufSize}};
        cl::Program program(context, devices, bins, NULL, &err);
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE |
                           CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        cl::Kernel krnl_ntt(program, "nantucket", &err);

        // Tell the kernel if this is a 2**18 or 2**24 sized problem.
        krnl_ntt.setArg(0, chicken_bits);

#endif

        // Create the buffers and allocate memory on the CPU side.
        const unsigned int point_size = sizeof(unsigned long);
        const unsigned int bytes_per_hbm_channel = (point_size * Nmax) / NLANE;

#if (USE_XRT==1)

	std::cout << "Doing xrt::bo() ..." << std::endl;
        std::vector<xrt::bo> pt_buf;
        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
          pt_buf.push_back(xrt::bo(device, bytes_per_hbm_channel, krnl.group_id(ch+1)));
        }

	std::cout << "Doing xrt::bo::map() ..." << std::endl;
        unsigned long *p_pt_buf[2*NLANE];
        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
          p_pt_buf[ch] = pt_buf[ch].map<unsigned long*>();
        }

#else

	std::vector<cl::Buffer> pt_buf;
        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
          pt_buf.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, bytes_per_hbm_channel, NULL, &err));
        }

        // Tell XRT which buffers will be used by which kernels so that allocations can
        // be performed in the correct HBM
        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
          krnl_ntt.setArg(ch+1, pt_buf[ch]);
        }

        // Schedule transfer of inputs to device memory, execution of kernel,
        // and transfer of outputs back to host memory
        cl::vector<cl::Memory> pt_buf_vec;
        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
          pt_buf_vec.push_back(pt_buf[ch]);
        }

        // Map host-side buffer memory to user-space pointers
        unsigned long *p_pt_buf[2*NLANE];
        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
          p_pt_buf[ch] = (unsigned long *)q.enqueueMapBuffer(pt_buf[ch], CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, bytes_per_hbm_channel);
        }

#endif

        // Initialize the vectors used in the test
        for (unsigned int p = 0; p < Nmax; p++) {

          const unsigned int ch  = get_channel_from_point(p)*2; // *2 to map to HBM channels
          const unsigned int idx = get_index_from_point(p, 0);
          unsigned long val = 0;

          if (p < Npts) {
            // Convert from hex string to unsigned long.
            std::stringstream str(input_pts[p]);
            str >> std::hex >> val;
          }

          p_pt_buf[ch+0][idx] = val;
          p_pt_buf[ch+1][idx] = 0;
        }

        /*

                Code body
                Insert your Initialization(e.g copying data to HBM etc.) host code before executing the kernel
         */

#if (USE_XRT==1)

	// Synchronize buffer content with device side
	std::cout << "synchronize input buffer data to device global memory\n";
        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
	  pt_buf[ch].sync(XCL_BO_SYNC_BO_TO_DEVICE);
	}

#else

	// Send the input points from CPU memory to HBM memory.
        q.enqueueMigrateMemObjects(pt_buf_vec, 0 /* 0 means from host*/);
        q.finish();

#endif
	
        //Time Measurement Starts Here
        auto kernel_start = std::chrono::high_resolution_clock::now(); // note start time
        /*

                Execute your kernel here
                Note: Please also add the pre-processing(operations performed on input data) functions here if any

         */

	//wait_for_enter("\nPress ENTER to continue after setting up ILA trigger...");

#if (USE_XRT==1)

	std::cout << "Execution of the kernel\n";
	auto run = xrt::run(krnl);
	run.set_arg(0,chicken_bits);
	for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
	  run.set_arg(ch+1,pt_buf[ch]);
	}
	run.start();
	run.wait();

#else

        q.enqueueTask(krnl_ntt); // Execute the kernel.
        q.finish();

#endif

	auto kernel_end = std::chrono::high_resolution_clock::now();  // note end time
        kernel_time = std::chrono::duration<double>(kernel_end - kernel_start); // calculate the difference
        std::cout << "Kernel time: ";
        std::cout << kernel_time.count()*1000 << " ms" << std::endl; // print the time in milliseconds

        /*

                Comparison
                Compare the generated results with the expected data
                Also please save the generated results in file
        
                outputFile_Handler << std::hex <<output_pts[i] << std::endl;

                Replace output_pts[i] with your output variable/buffer/pointer
         */

#if (USE_XRT==1)

        for (unsigned int ch = 0*NLANE; ch < 2*NLANE; ch++) {
	  pt_buf[ch].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
	}

#else

	// Send the computed points from HBM memory to CPU memory.
        q.enqueueMigrateMemObjects(pt_buf_vec, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();

#endif
	
        bool match = true;
        std::cout << "Checking computed results against expected ..." << std::endl;
        for (unsigned int p = 0; p < Nmax; p++) {
          // Use Nhop to support NTT sizes smaller than 2**24.
          if ((p % Nhop) == 0) {
            const unsigned int ch  = get_channel_from_point(p)*2; // *2 to map to HBM channels
            const unsigned int idx = get_index_from_point(p, 1);
            const unsigned long x_act = p_pt_buf[ch][idx];
            outputFile_Handler << std::hex << x_act << std::endl;

            // Convert from hex string to unsigned long.
            unsigned long x_exp;
            std::stringstream str(expect_pts[p/Nhop]);
            str >> std::hex >> x_exp;

            if (x_act != x_exp) {
              fprintf(stdout, "Error: @(%06x), exp: %016lx, act: %016lx, ch: %2d, idx: %06x\n",
                      p/Nhop, x_exp, x_act, ch, idx);
              match = false;
            /*
            } else {
              fprintf(stdout, "Match: @(%06x), exp: %016lx, act: %016lx, ch: %2d, idx: %06x\n",
                      p/Nhop, x_exp, x_act, ch, idx);
            */
            }
          }
        }

        /*
        // Check first pass intermediate results.
        const unsigned long M = 0xffffffff00000001;
        std::cout << "Checking computed results againt expected ..." << std::endl;
        for (unsigned int p = 0; p < Nmax; p++) {
          // Use Nhop to support NTT sizes smaller than 2**24.
          if ((p % Nhop) == 0) {
            const unsigned int ch  = get_channel_from_point(p)*2+1;
            const unsigned int idx = get_index_from_point(p, 0);
            const unsigned long x_act = p_pt_buf[ch][idx];
            outputFile_Handler << std::hex << x_act << std::endl;

            // Convert from hex string to unsigned long.
            unsigned long x_exp;
            std::stringstream str(expect_pts[p/Nhop]);
            str >> std::hex >> x_exp;

            if ((x_act % M) != x_exp) {
              fprintf(stdout, "Error: @(%06x), exp: %016lx, act: %016lx, ch: %2d, idx: %06x\n",
                      p/Nhop, x_exp, (x_act % M), ch, idx);
              match = false;
            } else {
              fprintf(stdout, "Match: @(%06x), exp: %016lx, act: %016lx, ch: %2d, idx: %06x\n",
                      p/Nhop, x_exp, (x_act % M), ch, idx);
            }
          }
        }
        */

        std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
        return (match ? EXIT_SUCCESS : EXIT_FAILURE);

        outputFile_Handler.close();

#if (USE_XRT==1)
#else
        delete[] fileBuf;
#endif
}
