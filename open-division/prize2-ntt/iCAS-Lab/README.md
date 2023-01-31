# iCAS_NTT

## Preliminaries:

Download a new copy of the test input and test output for NTT. Get the absolute/global paths of `in` and `out` directories for adding to the code.

Before running you should change the following variables in the `ntt_2_24_opt/main.cpp` file.

<!-- ### `ntt/src/ntt_host.cpp`

- `const std::string IN_FILE_DATA = "/home/<YOURUSERNAME>/ntt_2_24/data/in/fully_random_2_24";`
- `const std::string IN_FILE_OMEGA = "/home/<YOURUSERNAME>/ntt_2_24/data/omega_powers.out";`
- `const std::string OUT_FILE = "/home/<YOURUSERNAME>/ntt_2_24/data/out/fully_random_2_24_reorder";` -->

### `ntt_2_24_opt/main.cpp`

- `const std::string IN_FILE_PATH = "/home/<YOURUSERNAME>/ntt_2_24_opt/data/data/in/";`
- `const std::string OUT_FILE_PATH = "/home/<YOURUSERNAME>/ntt_2_24_opt/data/out/";`
- `const std::string POWERS_FILE_PATH = "/home/<YOURUSERNAME>/ntt_2_24_opt/data/";`

Change the paths so that they point at the directores `in` and `out` which contain files named with the prefixes `linear_2_` or `fully_random_2_`.

## File/Directory Structure:

- `ntt` - Design source files.
- `ntt_2_24_opt`
- `ntt_kernels` - Host code and hardware kernel implementations for NTT.
- `ntt_system` - Hardware build files.
- `ntt_system_hw_link` - Hardware and host link build files.
- `sdx_export_metadata`

<!-- 1. Activate the environment by using the following commands.

```
################################################################################
# Xilinx Environment
################################################################################
# TODO: use `cat /tools/xilinx_bash_setup > ~/.bashrc` to setup
source /tools/Xilinx/Vitis/2022.1/settings64.sh
source /opt/xilinx/xrt/setup.sh
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
export XILINXD_LICENSE_FILE=2200@beta-icaslab.cse.sc.edu
```

2. Open Vitis HLS using `vitis_hls` in the command line.
3. Create a new project using the 'Alveo U55' board.
4. Click through using the default settings.
5. Right click on `Source` files and click `Add Source file`. Add all the base `*.cpp` and `*.hpp` files. Add the `arithmetic.cpp`, `arithmetic.hpp`, `ntt.cpp`, and `ntt.hpp` files here.
6. Now right click on the `Test Bench` option and select `Add Test Bench File`. Add the `arithmetic_test.cpp`, `ntt_test.cpp`, and the `main.cpp` files here.

# C Simulation

From this point on, you should be able to run the C Simulation by using the menu button beside the green play button at the top toolbar.

All tests should complete and print out `PASS`.

# C Synthesis

Now we can synthesize the code, but we need to specify the top-level function (i.e. the function to implement in hardware).

1. Select the `Project` menu and select `Project Settings`.
2. Select `Synthesis` in the left pane.
3. Set the `Top Function` as the EXACT spelling of the C/C++ function in your code. (i.e. `NTT_2_24_in_place`).

Note: You can also use the `Browse...` function to see all the functions you can choose from to be the top function.

Now you can use the `C Synthesis` option in the dropdown beside the green play button.

The synthesis should complete without and violations. If your synthesis completes without errors and has violations then you may need to adjust the `#pragma`(s) in your top function C/C++ code.

# Cosimulation

Once you have completed the C Synthesis you can run the cosimulation. The cosimulation first runs your C code implementation. Then it will attempt to simulate the C Synthesized RTL and check whether the test cases in the testbench(s) pass or not.

Your cosimulation should complete without error and all tests should pass. -->
