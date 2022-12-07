NVCC=nvcc
TARGET=test_fft
TARGET_FATBIN=fft.fatbin
TARGET_DEBUG_FATBIN=fft_debug.fatbin
SRCS := $(wildcard *.cu)
SRCS_BIN := $(filter-out fft.cu,$(SRCS))
SRCS_FATBIN := $(filter-out test_fft.cu,$(SRCS))
OBJS := $(SRCS_BIN:.cu=.o)
OBJS_FATBIN := $(SRCS_BIN:.cu=.release.ptx)
OBJS_DEBUG_FATBIN := $(SRCS_BIN:.cu=.debug.ptx)

INCLUDES =
CCFLAGS=--relocatable-device-code=true

IS_CUDA_11:=${shell expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80 86
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode=arch=compute_$(sm),code=sm_$(sm)))

NVCC_DEBUG_CCFLAGS=--device-c -G -O0 -g -rdc=true -x cu
NVCC_CCFLAGS=-rdc=true

.PHONY: all
all : ${TARGET} ptx fft.fatbin

%.o: %.cu
	$(NVCC) $(INCLUDES) $(NVCC_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(TARGET): $(OBJS)
	$(NVCC) $(CCFLAGS) $(GENCODE_FLAGS) -link -o $@ $+

%.release.ptx: %.cu
	$(NVCC) -ptx $(NVCC_CCFLAGS) $< -o $@

%.debug.ptx: %.cu
	$(NVCC) --device-debug -ptx $(NVCC_CCFLAGS) $< -o $@

ptx: $(OBJS_FATBIN) $(OBJS_DEBUG_FATBIN)

$(TARGET_FATBIN): $(SRCS_FATBIN)
	$(NVCC) $(GENCODE_FLAGS) -dlink -fatbin $+ -o $@

$(TARGET_DEBUG_FATBIN): $(SRCS_FATBIN)
	$(NVCC) --device-debug $(GENCODE_FLAGS) -dlink -fatbin $+ -o $@

clean:
	rm -f ${TARGET} *.o *.fatbin *.ptx
