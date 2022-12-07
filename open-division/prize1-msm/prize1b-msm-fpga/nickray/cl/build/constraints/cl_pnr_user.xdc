# This contains the CL specific constraints for Top level PNR

create_pblock pblock_CL_top
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/CL_TST_DDR_A*}]
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_A_TST_AXI4_REG_SLC_2*}]
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/gen_ddr_tst[0].*}]
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_cores.DDR4_0*}]
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_inst[0].*}]
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_stat[0].*}]
resize_pblock [get_pblocks pblock_CL_top] -add {CLOCKREGION_X0Y10:CLOCKREGION_X5Y14}
set_property PARENT pblock_CL [get_pblocks pblock_CL_top]

create_pblock pblock_CL_mid
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/CL_TST_DDR_B*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells [list WRAPPER_INST/CL/CL_DMA_PCIS_SLV/CL_TST_DDR_B/CL_TST/sync_rst_n_reg]] -clear_locs
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_A_TST_AXI4_REG_SLC_1*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_B_TST_AXI4_REG_SLC_1*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_D_TST_AXI4_REG_SLC_1*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_B_TST_AXI4_REG_SLC_2*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/gen_ddr_tst[1].*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_cores.DDR4_1*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_inst[1].*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_stat[1].*}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/CL_TST_DDR_C}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_C_TST_AXI4_REG_SLC}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_C_TST_AXI4_REG_SLC_1}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/AXI_CROSSBAR}]
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_PCIM_MSTR/CL_TST_PCI}] -clear_locs
add_cells_to_pblock [get_pblocks pblock_CL_mid] [get_cells [list WRAPPER_INST/CL/CL_PCIM_MSTR/PCI_AXI4_REG_SLC]]
#resize_pblock [get_pblocks pblock_CL_mid] -add {CLOCKREGION_X0Y5:CLOCKREGION_X3Y9}
resize_pblock [get_pblocks pblock_CL_mid] -add {SLICE_X88Y300:SLICE_X107Y599}
resize_pblock [get_pblocks pblock_CL_mid] -add {DSP48E2_X11Y120:DSP48E2_X13Y239}
resize_pblock [get_pblocks pblock_CL_mid] -add {LAGUNA_X12Y240:LAGUNA_X15Y479}
resize_pblock [get_pblocks pblock_CL_mid] -add {RAMB18_X7Y120:RAMB18_X7Y239}
resize_pblock [get_pblocks pblock_CL_mid] -add {RAMB36_X7Y60:RAMB36_X7Y119}
resize_pblock [get_pblocks pblock_CL_mid] -add {URAM288_X2Y80:URAM288_X2Y159}
resize_pblock [get_pblocks pblock_CL_mid] -add {CLOCKREGION_X0Y5:CLOCKREGION_X2Y9}
set_property SNAPPING_MODE ON [get_pblocks pblock_CL_mid]
set_property PARENT pblock_CL [get_pblocks pblock_CL_mid]

create_pblock pblock_CL_bot
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/CL_TST_DDR_D*}]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/CL_DMA_PCIS_SLV/DDR_D_TST_AXI4_REG_SLC_2*}]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/gen_ddr_tst[2].*}]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_cores.DDR4_2*}]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_inst[2].*}]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -quiet -hierarchical -filter {NAME =~ WRAPPER_INST/CL/SH_DDR/ddr_stat[2].*}]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells [list WRAPPER_INST/CL/CL_DMA_PCIS_SLV/PCI_AXL_REG_SLC WRAPPER_INST/CL/CL_OCL_SLV/AXIL_OCL_REG_SLC WRAPPER_INST/CL/CL_SDA_SLV/AXIL_SDA_REG_SLC]]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -hierarchical -filter { NAME =~  "*CL/CL_OCL_SLV/slv_tst_wdata_reg[*][*]*" && PRIMITIVE_TYPE =~ REGISTER.*.* }]
#Reassign select cells to parent Pblock for better QoR
set pblock_cells [get_cells {WRAPPER_INST/CL/CL_DMA_PCIS_SLV/CL_TST_DDR_D/CL_TST/sync_rst_n_reg WRAPPER_INST/CL/CL_DMA_PCIS_SLV/CL_TST_DDR_D/CL_TST/pre_sync_rst_n_reg}]
add_cells_to_pblock [get_pblocks pblock_CL] $pblock_cells
#resize_pblock [get_pblocks pblock_CL_bot] -add {CLOCKREGION_X0Y0:CLOCKREGION_X3Y4}
resize_pblock [get_pblocks pblock_CL_bot] -add {SLICE_X88Y0:SLICE_X107Y299}
resize_pblock [get_pblocks pblock_CL_bot] -add {DSP48E2_X11Y0:DSP48E2_X13Y119}
resize_pblock [get_pblocks pblock_CL_bot] -add {LAGUNA_X12Y0:LAGUNA_X15Y239}
resize_pblock [get_pblocks pblock_CL_bot] -add {RAMB18_X7Y0:RAMB18_X7Y119}
resize_pblock [get_pblocks pblock_CL_bot] -add {RAMB36_X7Y0:RAMB36_X7Y59}
resize_pblock [get_pblocks pblock_CL_bot] -add {URAM288_X2Y0:URAM288_X2Y79}
resize_pblock [get_pblocks pblock_CL_bot] -add {CLOCKREGION_X0Y0:CLOCKREGION_X2Y4}
set_property SNAPPING_MODE ON [get_pblocks pblock_CL_bot]
set_property PARENT pblock_CL [get_pblocks pblock_CL_bot]

#set_clock_groups -name TIG_SRAI_1 -asynchronous -group [get_clocks -of_objects [get_pins static_sh/SH_DEBUG_BRIDGE/inst/bsip/inst/USE_SOFTBSCAN.U_TAP_TCKBUFG/O]] -group [get_clocks -of_objects [get_pins WRAPPER_INST/SH/kernel_clks_i/clkwiz_sys_clk/inst/CLK_CORE_DRP_I/clk_inst/mmcme3_adv_inst/CLKOUT0]]
#set_clock_groups -name TIG_SRAI_2 -asynchronous -group [get_clocks -of_objects [get_pins static_sh/SH_DEBUG_BRIDGE/inst/bsip/inst/USE_SOFTBSCAN.U_TAP_TCKBUFG/O]] -group [get_clocks drck]
#set_clock_groups -name TIG_SRAI_3 -asynchronous -group [get_clocks -of_objects [get_pins static_sh/SH_DEBUG_BRIDGE/inst/bsip/inst/USE_SOFTBSCAN.U_TAP_TCKBUFG/O]] -group [get_clocks -of_objects [get_pins static_sh/pcie_inst/inst/gt_top_i/diablo_gt.diablo_gt_phy_wrapper/phy_clk_i/bufg_gt_userclk/O]]









create_pblock pblock_CL_mid_left
resize_pblock [get_pblocks pblock_CL_mid_left] -add {CLOCKREGION_X0Y5:CLOCKREGION_X1Y9}
set_property PARENT pblock_CL_mid [get_pblocks pblock_CL_mid_left]



add_cells_to_pblock [get_pblocks pblock_CL_mid_left] [get_cells -hierarchical -filter {NAME =~ WRAPPER_INST/CL/top_inst/twisted_edwards_prek_full_inst/twisted_edwards_prek_inst/mul_mont_inst_R5_am*}]

add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -hierarchical -filter {NAME =~ WRAPPER_INST/CL/top_inst/twisted_edwards_prek_full_inst/twisted_edwards_prek_inst/mul_mont_inst_R6_am*}]

add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -hierarchical -filter {NAME =~ WRAPPER_INST/CL/top_inst/twisted_edwards_prek_full_inst/twisted_edwards_prek_inst/mul_mont_inst_R7_m*}]


add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -hierarchical -filter {NAME =~ WRAPPER_INST/CL/top_inst/twisted_edwards_prek_full_inst/twisted_edwards_prek_inst/mul_mont_inst_out0_x*}]
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -hierarchical -filter {NAME =~ WRAPPER_INST/CL/top_inst/twisted_edwards_prek_full_inst/twisted_edwards_prek_inst/mul_mont_inst_out0_y*}]
add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -hierarchical -filter {NAME =~ WRAPPER_INST/CL/top_inst/twisted_edwards_prek_full_inst/twisted_edwards_prek_inst/mul_mont_inst_out0_z*}]
add_cells_to_pblock [get_pblocks pblock_CL_top] [get_cells -hierarchical -filter {NAME =~ WRAPPER_INST/CL/top_inst/twisted_edwards_prek_full_inst/twisted_edwards_prek_inst/mul_mont_inst_out0_t*}]





add_cells_to_pblock [get_pblocks pblock_CL_bot] [get_cells -hierarchical -filter {NAME =~ get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR}]

set_property LOC URAM288_X0Y0 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y1 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y2 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y3 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y4 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y5 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y6 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y7 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[0].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y8 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y9 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y10 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y11 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y12 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y13 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y14 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y15 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[1].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y16 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y17 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y18 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y19 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y20 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y21 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y22 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y23 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[2].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y24 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y25 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y26 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y27 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y28 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y29 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y30 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y31 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[3].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y32 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y33 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y34 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y35 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y36 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y37 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y38 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y39 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[4].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y40 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y41 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y42 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y43 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y44 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y45 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y46 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y47 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[5].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y48 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y49 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y50 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y51 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y52 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y53 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y54 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y55 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[6].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y56 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y57 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y58 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y59 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y60 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y61 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y62 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y63 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[7].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y64 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y65 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y66 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y67 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y68 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y69 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y70 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y71 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[8].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y72 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y73 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y74 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y75 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y76 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y77 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y78 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X0Y79 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[9].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y0 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y1 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y2 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y3 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y4 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y5 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y6 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y7 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[10].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y8 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y9 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y10 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y11 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y12 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y13 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y14 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y15 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[11].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y16 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y17 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y18 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y19 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y20 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y21 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y22 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y23 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[12].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y24 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y25 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y26 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y27 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y28 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y29 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y30 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y31 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[13].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y32 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y33 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y34 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y35 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y36 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y37 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y38 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y39 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[14].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y40 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y41 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y42 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y43 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y44 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y45 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y46 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y47 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[15].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y48 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[0].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y49 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[1].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y50 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[2].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y51 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[3].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y52 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[4].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y53 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[5].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y54 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[6].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
set_property LOC URAM288_X1Y55 [get_cells WRAPPER_INST/CL/top_inst/G_I[0].bucket_storage_inst/SLR.G_C[16].G_R[7].ram_inst/xpm_memory_sdpram_inst/xpm_memory_base_inst/gen_wr_a.gen_word_narrow.mem_reg_uram_0]
