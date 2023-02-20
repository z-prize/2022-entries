############################## Help Section ##############################
.PHONY: help
help:
	@echo "Makefile Usage:"
	@echo "  make clean"
	@echo "      Command to remove the generated non-hardware files."
	@echo ""
	@echo "  make cleanpurge"
	@echo "      Command to remove all the generated files."
	@echo ""
	@echo  "  make test"
	@echo  "     Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	@echo  ""
	@echo "  make run TARGET=<sw_emu/hw_emu/hw> "
	@echo "      Command to build and run xclbin application."
	@echo ""
	@echo "  make build TARGET=<sw_emu/hw_emu/hw>"
	@echo "      Command to build xclbin application."
	@echo ""
	@echo "  make host"
	@echo "      Command to build host application."
	@echo ""
	@echo "  make testvectors"
	@echo "      Command to generate testvectors."
	@echo ""

############################## Setting up Project Variables ##############################

include ./mk/config.mk

BUILD_DIR := ./build/$(PROJECT)/$(PLATFORM)/$(TARGET)
RUN_DIR := ./run/$(PROJECT)/$(PLATFORM)/$(TARGET)

TMP_DIR := $(BUILD_DIR)/tmp
LOG_DIR := $(BUILD_DIR)/logs
REP_DIR := $(BUILD_DIR)/reports

$(shell mkdir -p $(RUN_DIR))
$(shell mkdir -p $(TMP_DIR))
$(shell mkdir -p $(LOG_DIR))
$(shell mkdir -p $(REP_DIR))

############################## Setting up Host Variables ##############################

# OpenCL flags : includes XRT/include, VIVADO/include
CXXFLAGS += -I$(XILINX_XRT)/include -I$(XILINX_VIVADO)/include -I$(XILINX_HLS)/include -Wall -O3 -g -std=c++1y
LDFLAGS += -L$(XILINX_XRT)/lib -lOpenCL -pthread 

# Include Required Host Source Files
CXXFLAGS += -Icommon/includes/cmdparser -Icommon/includes/logger -Isrc/hls
HOST_SRCS += common/includes/cmdparser/cmdlineparser.cpp common/includes/logger/logger.cpp 
HOST_SRCS += src/host/*

# Host compiler global settings
CXXFLAGS += -fmessage-length=0
LDFLAGS += -lrt -lstdc++ 
LDFLAGS += -luuid -lxrt_coreutil

############################## Setting up Kernel Variables ##############################

# Kernel compiler global settings
$(shell printf "[hls]\nclock=$(FCLK)000000:ntt_2_24" > cfg/clock.cfg) 
CONFIG_FILES := $(patsubst %,--config %, $(wildcard cfg/*.cfg))

VPP_FLAGS += --target $(TARGET) --platform $(PLATFORM) $(CONFIG_FILES) --save-temps --kernel_frequency $(FCLK) --optimize 3 -D$(NTT_2_12_FLAG)
VPP_FLAGS += --temp_dir $(TMP_DIR) --log_dir $(LOG_DIR) --report_dir $(REP_DIR)

ifneq ($(TARGET), hw)
	VPP_FLAGS += -g
endif

############################## Setting Targets ##############################

KERNELS := ntt_2_$(log2N).xo
KERNELS := $(patsubst %, $(BUILD_DIR)/%, $(KERNELS))
BINARY_CONTAINER := $(BUILD_DIR)/$(PROJECT).xclbin
EXECUTABLE := $(RUN_DIR)/$(PROJECT)

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: $(BINARY_CONTAINER)

.PHONY: kernels
kernels: $(KERNELS)

# Generating .xo
$(BUILD_DIR)/%.xo : src/hls/*.cpp src/hls/*.hpp   
	v++ $(VPP_FLAGS) --compile --kernel $* -o $@ $^ $(XCL_EMULATION_MODE)

# Building kernel
$(BINARY_CONTAINER): $(KERNELS)
	v++ $(VPP_FLAGS) --link $(VPP_LDFLAGS) -o $@ $^ 

############################## Setting Rules for Host (Building Host Executable) ##############################

$(EXECUTABLE): $(HOST_SRCS) 
		$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

############################## Setting Essential Checks and Running Rules ##############################

RUN_APP :=
ifeq ($(TARGET),hw)
	RUN_APP := ./$(PROJECT) --xclbin_file $(PROJECT).xclbin --device_id $(DEVICE_ID) --log2N $(log2N)
else
	RUN_APP := XCL_EMULATION_MODE=$(TARGET) ./$(PROJECT) --xclbin_file $(PROJECT).xclbin --device_id $(DEVICE_ID) --log2N $(log2N)
endif

.PHONY: run _run
run : build _run
# use _run if you don't want to trigger a build
_run: host 
	cp $(BINARY_CONTAINER) $(RUN_DIR)
	cp cfg/xrt.ini $(RUN_DIR)
	cp scripts/xsim.tcl $(RUN_DIR)
	cp testvectors/* $(RUN_DIR)
	cd $(RUN_DIR); \
	emconfigutil --platform $(PLATFORM); \
	$(RUN_APP)

############################## Utility ##############################

.PHONY: platforminfo
platforminfo: 
	platforminfo --force --platform $(PLATFORM) -o $(PLATFORM).info

.PHONY: vitis_analyzer
vitis_analyzer: 
	vitis_analyzer .

############################## Testvectors ##############################

.PHONY: testvectors
testvectors:
	cd testvectors; \
	python3 testvectors.py

############################## Cleaning Rules ##############################

.PHONY: clean
clean:
	-$(RM) *.jou *.log *.str

.PHONY: cleanhls
cleanhls: clean
	-$(RM) -r .hls.* 

.PHONY: cleanrun
cleanrun: clean
	-$(RM) -r run/*

.PHONY: cleanbuild
cleanbuild: clean
	-$(RM) -r build/*

.PHONY: cleanpurge
cleanpurge: cleanrun cleanbuild cleanhls
	-$(RM) -r .Xil .run .ipcache

.PHONY: cleanthisbuild
cleanthisbuild: clean
	-$(RM) -r $(BUILD_DIR)