# PROJECT settings

# log2N = 24, other sizes not supported
log2N := 24
# NTT_2_12_FLAG := {NTT_2_12_DATAFLOW, NTT_2_12_SERIAL}
# FCLK := {100, 200}
# NTT_2_12_FLAG := NTT_2_12_DATAFLOW
# FCLK := 100
NTT_2_12_FLAG := NTT_2_12_SERIAL
FCLK := 200
# PLATFORM := {xilinx_u280_gen3x16_xdma_1_202211_1, xilinx_u55n_gen3x4_xdma_2_202110_1}, both work without changes
PLATFORM := xilinx_u55n_gen3x4_xdma_2_202110_1
DEVICE_ID := 1
# TARGET := {sw_emu|hw_emu|hw}, hw_emu will run for an extremely long time
TARGET := hw

# Project
PROJECT := COSIC_NTT_2_$(log2N)_$(FCLK)_MHz_$(NTT_2_12_FLAG)