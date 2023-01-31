# Copyright Supranational LLC
# Licensed under the Apache License, Version 2.0, see LICENSE-APACHE 
# or the MIT license, see LICENSE-MIT, at your option.
# SPDX-License-Identifier: Apache-2.0 OR MIT

# Example use cases...
#
# Run HW emulation assuming input/expects filenames in main.cpp are valid:
# make run TARGET=hw_emu
#
# Build and run HW target with XRT API assuming input/expects filenames in main.cpp are valid:
# make run
#
# Build and run HW target with OpenCL API assuming input/expects filenames in main.cpp are valid:
# make run_opencl
#
# Build and run with XRT API and specifying input/expected points on the command line:
# make build
# cd hw; ./app.exe <input-points-filename> <expected-points-filename>

TARGET    := hw
PLATFORM  := xilinx_u55n_gen3x4_xdma_2_202110_1
BUILD_DIR := $(TARGET)
RTL_DIR   := ../rtl

CXXFLAGS  += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -Wall -O0 -g -std=c++1y -DUSE_XRT=1
LDFLAGS   += -L$(XILINX_XRT)/lib -pthread -lOpenCL

# Host compiler global settings
CXXFLAGS  += -fmessage-length=0
LDFLAGS   += -lrt -lstdc++ 
LDFLAGS   += -luuid -lxrt_coreutil

XRT_CFILES =	main.cpp \
		handler_function.cpp

OPENCL_CFILES = $(XRT_CFILES) util.cpp

CFILES =	$(OPENCL_CFILES) ntt_cfg.h util.h handler_function.h

VFILES  = $(addprefix $(RTL_DIR)/,$(shell cat $(RTL_DIR)/files.f))
TCLARGS = $(subst $() $(),:,$(addprefix ../,$(VFILES)))

run: build
ifeq ($(TARGET),hw)
	cp xrt.ini $(BUILD_DIR)
	cd $(BUILD_DIR) && ./app.exe
else
	cp xrt.ini $(BUILD_DIR)
	cd $(BUILD_DIR) && XCL_EMULATION_MODE=$(TARGET) ./app.exe
endif

run_opencl: build
ifeq ($(TARGET),hw)
	cp xrt.ini $(BUILD_DIR)
	cd $(BUILD_DIR) && ./app_opencl.exe
else
	cp xrt.ini $(BUILD_DIR)
	cd $(BUILD_DIR) && XCL_EMULATION_MODE=$(TARGET) ./app_opencl.exe
endif

build: $(BUILD_DIR)/app.exe  $(BUILD_DIR)/app_opencl.exe $(BUILD_DIR)/emconfig.json $(BUILD_DIR)/nantucket.xclbin

$(BUILD_DIR)/app.exe: $(XRT_CFILES) $(BUILD_DIR)/nantucket.xclbin
	mkdir -p $(BUILD_DIR)
	g++ -o $@ $(XRT_CFILES) $(CXXFLAGS) $(LDFLAGS)

$(BUILD_DIR)/app_opencl.exe: $(OPENCL_CFILES) $(BUILD_DIR)/nantucket.xclbin
	mkdir -p $(BUILD_DIR)
	g++ -Wall -g -std=c++11 $(OPENCL_CFILES) -o $@ \
		-I. -Ikernel \
		-I${XILINX_XRT}/include/ \
		-I/tools/Xilinx/Vitis_HLS/2022.1/include/ \
		-L${XILINX_XRT}/lib/ -lOpenCL -pthread -lrt -lstdc++

kernel/nantucket.xo: kernel/gen_xo.tcl kernel/package_kernel.tcl $(CFILES) $(VFILES)
	cd $(<D); vivado -nojournal -mode batch -source $(<F) -tclargs $(TCLARGS)

$(BUILD_DIR)/nantucket.xclbin: kernel/nantucket.xo ./u55n.cfg
	v++ -R2 -l -t ${TARGET} --platform $(PLATFORM) --config ./u55n.cfg  --save-temps \
	--kernel_frequency "0:300|1:500" \
	--vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=./kernel/pr.xdc $< -o $@ 

$(BUILD_DIR)/emconfig.json:
	emconfigutil --platform $(PLATFORM) --od $(BUILD_DIR) --nd 1

clean:
	-rm -rf sw_emu hw_emu hw *json opencl* *log *summary _x xilinx* .run .Xil .ipcache *.jou
	-rm -f kernel/nantucket.xo
	-rm -fr kernel/packaged_kernel*
	-rm -fr kernel/tmp_kernel_pack*
	-rm -f kernel/vivado.log kernel/vivado.jou kernel/vivado*.backup.{log,jou}
