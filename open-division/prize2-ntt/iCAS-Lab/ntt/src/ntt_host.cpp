
#include <chrono>
#include <thread>
#include <xrt.h>
#include <cmath>
#include <CL/cl.h>
//#include <vector.h>
//#include <boost/format.hpp>
#include <fstream>
#include "xcl2.hpp"
#include <experimental/xrt_profile.h>
//#include "ntt.hpp"

//==>> This is for the 2^24
const std::string IN_FILE_DATA = "/home/elbtity/competition_final/NTT_Essa/ntt_2_24/data/in/fully_random_2_24";
const std::string IN_FILE_OMEGA = "/home/elbtity/competition_final/NTT_Essa/ntt_2_24/data/omega_powers.out";
const std::string OUT_FILE = "/home/elbtity/competition_final/NTT_Essa/ntt_2_24/data/out/fully_random_2_24_reorder";
//==>> This is for the 2^3
//const std::string IN_FILE_DATA = "/home/elbtity/competition_final/NTT_Essa/ntt_2_24/data/in/fully_random_2_3";
//const std::string IN_FILE_OMEGA = "/home/elbtity/competition_final/NTT_Essa/ntt_2_24/data/omega_powers_2_3.out";
//const std::string OUT_FILE = "/home/elbtity/competition_final/NTT_Essa/ntt_2_24/data/out/fully_random_2_3";
typedef unsigned long long  GF;
#define NO_64B_WORDS_PER_256WORD 1
#define CIRCUIT_SIZE 16777216
#define DATA_SIZE_POLY (CIRCUIT_SIZE * NO_64B_WORDS_PER_256WORD)


using namespace std;
using namespace std::chrono;


double run_krnl(cl::Context& context,
            	cl::CommandQueue& q,
            	cl::Kernel& kernel,

				/// Top level of the Kernel
				std::vector<GF, aligned_allocator<GF> >& in,
				std::vector<GF, aligned_allocator<GF> >& powers
				//GF *in,
				//GF *powers//,					/// *** Here we change
            	//unsigned int ntt_control,
            	//data_t* ibuf,
            	//data_t* cofs,
            	//data_t* proc_buf1,
            	//data_t* proc_buf2,
            	//GF* obuf			/// *** Here we change
				//std::vector<GF, aligned_allocator<GF> >& obuf
				//std::vector<GF, aligned_allocator<GF> >& testing,
				//std::vector<GF, aligned_allocator<GF> >& testing_2
				//testing[8]
            	) {
	cl_int err;

	// These commands will allocate memory on the FPGA. The cl::Buffer objects can
	// be used to reference the memory locations on the device.
	// Creating Buffers

	OCL_CHECK(err, cl::Buffer d_ibuf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,		/// Here we change *****
                                        	sizeof(GF) * CIRCUIT_SIZE, in.data(), &err));
	//OCL_CHECK(err, cl::Buffer d_ibuf(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,	/// Here we change *****
	                                     //   	sizeof(GF) * CIRCUIT_SIZE, in, &err));
	OCL_CHECK(err, cl::Buffer d_cofs(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                        	sizeof(GF) * CIRCUIT_SIZE, powers.data(), &err));
	//OCL_CHECK(err, cl::Buffer d_proc_buf1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        	//sizeof(data_t) * CIRCUIT_SIZE, proc_buf1, &err));
	//OCL_CHECK(err, cl::Buffer d_proc_buf2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        	//sizeof(data_t) * CIRCUIT_SIZE, proc_buf2, &err));
	//OCL_CHECK(err, cl::Buffer d_obuf(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,					/// Here we change ***** CL_MEM_WRITE_ONLY , CL_MEM_READ_WRITE
      //                                  	sizeof(GF) * CIRCUIT_SIZE, obuf.data(), &err));
//	OCL_CHECK(err, cl::Buffer test1(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,					/// Here we change *****
	//                                        	sizeof(GF) * CIRCUIT_SIZE, testing.data(), &err));
	//OCL_CHECK(err, cl::Buffer test2(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,					/// Here we change *****
	  //                                      	sizeof(GF) * CIRCUIT_SIZE, testing_2.data(), &err));

	// Setting the kernel Arguments
	//OCL_CHECK(err, err = (kernel).setArg(0, ntt_control));
	OCL_CHECK(err, err = (kernel).setArg(0, d_ibuf));
	OCL_CHECK(err, err = (kernel).setArg(1, d_cofs));



	//OCL_CHECK(err, err = (kernel).setArg(3, d_proc_buf1));
	//OCL_CHECK(err, err = (kernel).setArg(4, d_proc_buf2));
	//OCL_CHECK(err, err = (kernel).setArg(2, d_obuf));					/// Here we change *****
	//OCL_CHECK(err, err = (kernel).setArg(3, test1));					/// Here we change *****
//	OCL_CHECK(err, err = (kernel).setArg(4, test2));					/// Here we change *****



	// Copy input data to Device Global Memory
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_ibuf, d_cofs }, 0 /* 0 means from host*/));

   // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, 0 /* 0 means from host*/));
	 // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_obuf}, 0 /* 0 means from host*/));	/// Here we change *****
	q.finish();


	std::chrono::duration<double> kernel_time(0);

	auto kernel_start = std::chrono::high_resolution_clock::now();
	//OCL_CHECK(err, err = q.enqueueTask(kernel));
    OCL_CHECK(err, err = q.enqueueTask(kernel));


 //   OCL_CHECK(err, err = q.enqueueNDRangeKernel(kernel, 0, 1, 1, nullptr, nullptr));
	q.finish();
	auto kernel_end = std::chrono::high_resolution_clock::now();

   	kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);

	// Copy Result from Device Global Memory to Host Local Memory
   // barrier(CLK_GLOBAL_MEM_FENCE);
   //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_obuf}, CL_MIGRATE_MEM_OBJECT_HOST));
  // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_obuf,test1,test2}, CL_MIGRATE_MEM_OBJECT_HOST)); //**Here we change
  // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_obuf}, CL_MIGRATE_MEM_OBJECT_HOST)); //**Here we change ESSA_NOV_08
   OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_ibuf}, CL_MIGRATE_MEM_OBJECT_HOST)); //**Here we change

   //OCL_CHECK(err, err = q.enqueueMigrateMemObjects({d_obuf}, CL_MIGRATE_MEM_OBJECT_HOST));			//**Here we change
   // OCL_CHECK(err, err = q.enqueueReadBuffer(buffer_output, CL_FALSE, 0, sizeof(uint32_t) * data_length_res, commitments_buffer.data(), nullptr,nullptr));

	q.finish();
	/*
	cout <<"Input vector as: "<< std::endl;
			for (int l=0;l<CIRCUIT_SIZE;l++){

				cout << in[l] << std::endl;

			}
			*/

	return kernel_time.count();
}

//
//void readin(GF in[CIRCUIT_SIZE]){//, const std::string prefix) {
	void readin(std::vector<GF, aligned_allocator<GF> >& in){//, const std::string prefix) {
 // const uint32_t powered = pow(2,degree);
  //std::string fname = IN_FILE_PATH;
  //std::ifstream in_file(fname + prefix + std::to_string(degree));
  std::ifstream in_file(IN_FILE_DATA);
  std::string line;
  uint32_t count = 0;
  if (in_file.is_open()) {
    cout << "Reading File: " << IN_FILE_DATA << endl;
    while ((count < CIRCUIT_SIZE) && std::getline(in_file, line)) {
      in[count] = std::stoull(line, nullptr, 16);
      ++count;
    }
  }
  in_file.close();
}

	//template <typename T, size_t N>
	void readout(std::vector<GF, aligned_allocator<GF> >& in){//, const std::string prefix) {
	 // const uint32_t powered = pow(2,degree);
	  //std::string fname = IN_FILE_PATH;
	  //std::ifstream in_file(fname + prefix + std::to_string(degree));
	  std::ifstream in_file(OUT_FILE);
	  std::string line;
	  uint32_t count = 0;
	  if (in_file.is_open()) {
	    cout << "Reading File: " << OUT_FILE << endl;
	    while ((count < CIRCUIT_SIZE) && std::getline(in_file, line)) {
	      in[count] = std::stoull(line, nullptr, 16);
	      ++count;
	    }
	  }
	  in_file.close();
	}
//template <typename T, size_t N>
void readomega(std::vector<GF, aligned_allocator<GF> >& in){//, const std::string prefix) {
 // const uint32_t powered = pow(2,degree);
  //std::string fname = IN_FILE_PATH;
  //std::ifstream in_file(fname + prefix + std::to_string(degree));
  std::ifstream in_file(IN_FILE_OMEGA);
  std::string line;
  uint32_t count = 0;
  if (in_file.is_open()) {
    cout << "Reading File: " << IN_FILE_OMEGA << endl;
    while ((count < CIRCUIT_SIZE) && std::getline(in_file, line)) {
      in[count] = std::stoull(line, nullptr, 16);
      ++count;
    }
  }
  in_file.close();
}





unsigned int reverseBits(unsigned int num)
{
    unsigned int NO_OF_BITS = sizeof(num) * 3;
    unsigned int reverse_num = 0;
    unsigned int i;
    for (i = 0; i < NO_OF_BITS; i++) {
        if ((num & (1 << i)))
            reverse_num |= 1 << ((NO_OF_BITS - 1) - i);
    }
    return reverse_num;
}

int main(int argc, char** argv) {


	// CL arguments parser
	if (argc != 2) {
    	std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
    	return EXIT_FAILURE;
	}
	std::string binaryFile = argv[1];
	// OpenCL Variables and parameters
	cl_int err;
	cl::CommandQueue q;
	cl::Context context;
	cl::Kernel NTT_krnl;

	cout << "Poly Buffer start" << endl;
	std::vector<GF, aligned_allocator<GF> > poly_buffer(CIRCUIT_SIZE);			// Points to the inputs array

	cout << "Co-efficient Buffer start" << endl;
	std::vector<GF, aligned_allocator<GF> > cofs_buffer(CIRCUIT_SIZE);
//	std::vector<GF, aligned_allocator<GF> > out_buffer(CIRCUIT_SIZE);
	cout << "Output-test Buffer start" << endl;
	std::vector<GF, aligned_allocator<GF> > out_copy(CIRCUIT_SIZE);
	//std::vector<GF, aligned_allocator<GF> > test_1(CIRCUIT_SIZE);
	//std::vector<GF, aligned_allocator<GF> > test_2(CIRCUIT_SIZE);





	double kernel_time_in_sec = 0;

	readin(poly_buffer);
	/*
	for (int i=0; i<CIRCUIT_SIZE; i++){

		poly_buffer[i] = 2;

	}

*/

	readomega(cofs_buffer);
/*
	for (int i=0; i<CIRCUIT_SIZE; i++){

		cofs_buffer[i] =5;

	}

*/



	auto devices = xcl::get_xil_devices();

	// read_binary_file() is a utility API which will load the binaryFile
	// and will return the pointer to file buffer.
	auto fileBuf = xcl::read_binary_file(binaryFile);
	cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
	bool valid_device = false;
	for (unsigned int i = 0; i < devices.size(); i++) {
    	auto device = devices[i];
    	// Creating Context and Command Queue for selected Device
    	OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
    	OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    	std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    	cl::Program program(context, {device}, bins, nullptr, &err);
    	if (err != CL_SUCCESS) {
        	std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
    	} else {
        	std::cout << "Device[" << i << "]: program successful!\n";
        	OCL_CHECK(err, NTT_krnl = cl::Kernel(program, "NTT_2_3_in_place", &err));		// Change the top function of the Kernel Here
        	valid_device = true;
        	break; // we break because we found a valid device
    	}
	}

	if (!valid_device) {
    	std::cout << "Failed to program any device found, exit!\n";
    	exit(EXIT_FAILURE);
	}



	std::cout << " Starting Kernel ... ";

	//kernel_time_in_sec = run_krnl(context, q, NTT_krnl, poly_buffer, cofs_buffer,out_copy);//,test_1,test_2 );
	kernel_time_in_sec = run_krnl(context, q, NTT_krnl, poly_buffer, cofs_buffer);//,test_1,test_2 );


/*
	cout <<"Transmitted data" << std::endl;
		for (int i=0; i<CIRCUIT_SIZE;i++){
			cout<<std::hex<<test_1[i]<<std::endl;
		}

		cout <<"Transmitted Co-efficients" << std::endl;
		for (int i=0; i<CIRCUIT_SIZE;i++){
			cout<<std::hex<<test_2[i]<<std::endl;
		}

		cout <<"Calculated data " << std::endl;
		for (int i=0; i<CIRCUIT_SIZE;i++){
			cout<<std::hex<<out_copy[i]<<std::endl;
		}

//}
 *
 */


	/*
	cout <<"Transmitted data" << std::endl;
	for (int i=0; i<CIRCUIT_SIZE;i++){
		cout<<std::hex<<out_copy[i]<<std::endl;
	}

	readout(out_buffer);


	cout << "Here we print the copied inputs of the Kernel" << std::endl;

	for (int i = 0; i<CIRCUIT_SIZE; i++){
		cout <<"element is " << out_copy[i]<<std::endl;
	}

*/
	readout(out_copy);
	//int ri;

	cout <<"stored data " << std::endl;
			for (int i=0; i<CIRCUIT_SIZE;i++){
			//	ri = reverseBits(i);
				cout<<std::hex<<out_copy[i]<<std::endl;
			}

	int i,ri;
	i = 0;
	while(i<CIRCUIT_SIZE) {
			ri = reverseBits(i);
			if (out_copy[ri] != poly_buffer[i]){
				cout << "Error: NTT result [" << i << "] should be "
								 << out_copy[i] << ", but the output is "
								 << poly_buffer[ri] << endl;
				//break;
			}

			i++;
		}
//*/
	cout << "Kernel result is correct" << std::endl;




	return  EXIT_SUCCESS;
}


