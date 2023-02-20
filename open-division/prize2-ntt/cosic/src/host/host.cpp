#include "cmdlineparser.h"
#include <fstream>
#include <iostream>
#include <cstring>

// XRT includes
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "experimental/xrt_xclbin.h"
#include "experimental/xrt_ip.h"

// User includes
#include "ntt.hpp"
#include "timer.hpp"

using namespace std;

int main(int argc, char **argv)
{
    // Simplified host code for a single N = 2^24
    int N = (1 << 24);

    // Timer
    Timer timer;
    long long stopTime;

    // Command Line Parser
    sda::utils::CmdLineParser parser;
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.addSwitch("--log2N", "-n", "log2 of the NTT size", "24");
    parser.parse(argc, argv);
    if (argc < 3)
    {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    int device_index = stoi(parser.value("device_id"));
    cout << "[INFO] Open device[" << device_index << "]" << endl;
    auto device = xrt::device(device_index);

    string binaryFile = parser.value("xclbin_file");
    cout << "[INFO] Load xclbin: " << binaryFile << endl;
    auto uuid = device.load_xclbin(binaryFile);

    string krnl_name = "ntt_2_24";
    cout << "[INFO] Fetch compute kernel: " << krnl_name << endl;
    auto kernel = xrt::kernel(device, uuid, krnl_name, xrt::kernel::cu_access_mode::exclusive);

    size_t vector_size_bytes_per_hbm = sizeof(u64) * N / 16;
    cout << "[INFO] Allocate Buffer of size " << vector_size_bytes_per_hbm << "B in each HBM" << endl;
    xrt::bo bo_hbm[32];
    for (int i = 0; i < 32; i++)
    {
        // NTT 2^24 pass 1: HBM[16-32] -> HBM[0-15]
        // NTT 2^24 pass 2: HBM[0-15] -> HBM[16-32]
        // -> in/out is in HBM[16-32]. HBM[0-15] is device_only
        auto flag = (i < 16) ? xrt::bo::flags::device_only : xrt::bo::flags::normal;
        bo_hbm[i] = xrt::bo(device, vector_size_bytes_per_hbm, flag, kernel.group_id(i));
    }

    cout << "[INFO] Loading testvectors" << endl;
    string lineIn;
    string lineOut;
    ifstream fin("in_fully_random_2_24.txt");
    ifstream fout("out_fully_random_2_24.txt");
    if (fin.fail() || fout.fail())
    {
        throw runtime_error("Testvectors not found. Generate with `make testvectors`.");
        return EXIT_FAILURE;
    }
    u64 *expected_out = new u64[N];
    u64 *inputs = new u64[N];
    for (int i = 0; i < _2_24; i++)
    {
        getline(fin, lineIn);
        getline(fout, lineOut);
        inputs[i] = stoull(lineIn);
        expected_out[i] = stoull(lineOut);
    }

    cout << "[INFO] Synchronize input buffer data to device HBM";
    for (int i = 0; i < 16; i++)
    {
        bo_hbm[16 + i].write(&inputs[i * N / 16]);
    }
    TIMEFOR(for (int i = 16; i < 32; i++) { bo_hbm[i].sync(XCL_BO_SYNC_BO_TO_DEVICE); });

    cout << "[INFO] NTT IP Run ";
    TIME(auto run = kernel(bo_hbm[0], bo_hbm[1], bo_hbm[2], bo_hbm[3], bo_hbm[4], bo_hbm[5], bo_hbm[6], bo_hbm[7], bo_hbm[8], bo_hbm[9], bo_hbm[10], bo_hbm[11], bo_hbm[12], bo_hbm[13], bo_hbm[14], bo_hbm[15], bo_hbm[16], bo_hbm[17], bo_hbm[18], bo_hbm[19], bo_hbm[20], bo_hbm[21], bo_hbm[22], bo_hbm[23], bo_hbm[24], bo_hbm[25], bo_hbm[26], bo_hbm[27], bo_hbm[28], bo_hbm[29], bo_hbm[30], bo_hbm[31]); run.wait());

    cout << "[INFO] Synchronize output buffer data from device HBM";
    TIMEFOR(for (int i = 16; i < 32; i++) { bo_hbm[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE); });

    cout << "[INFO] Validating results" << endl;
    for (int i = 0; i < 16; i++)
    {
        if (memcmp(bo_hbm[16 + i].map<u64 *>(), &expected_out[i * N / 16], vector_size_bytes_per_hbm))
        {
            throw runtime_error("Value read back does not match reference");
            return EXIT_FAILURE;
        }
    }

    cout << "[INFO] TEST PASSED\n";
    delete[] expected_out;
    delete[] inputs;
    return EXIT_SUCCESS;
}
