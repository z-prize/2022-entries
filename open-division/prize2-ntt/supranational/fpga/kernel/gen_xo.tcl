set xoname    nantucket.xo
set krnl_name nantucket
set device    xilinx_u55n_gen3x4_xdma_2_202110_1

set suffix "${krnl_name}_${device}"
set path_to_packaged "./packaged_kernel_${suffix}"
set path_to_tmp_project "./tmp_kernel_pack_${suffix}"

create_project -force kernel_pack $path_to_tmp_project

if { $argc != 1 } {
    puts "The script requires a target and a colon separated file list."
    puts "Please try again."
} else {
    set vfiles [split [lindex $argv 0] ":"]
}

puts $vfiles
add_files $vfiles \
    ../../rtl/ntt/TWIDDLE_ROM_WA0_NLEVEL7.mem \
    ../../rtl/ntt/TWIDDLE_ROM_WA1_NLEVEL7.mem \
    ../../rtl/ntt/TWIDDLE_ROM_WA0_NLEVEL9.mem \
    ../../rtl/ntt/TWIDDLE_ROM_WA1_NLEVEL9.mem \
    ../../rtl/ntt/TWIDDLE_ROM_WA0_NLEVEL12.mem \
    ../../rtl/ntt/TWIDDLE_ROM_WA1_NLEVEL12.mem


update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

ipx::package_project -root_dir $path_to_packaged -vendor supranational.net -library RTLKernel -taxonomy /KernelIP -import_files -set_current true


# Kernel specific setup output from GUI flow.
source package_kernel.tcl
edit_core [ipx::current_core]

set_property ipi_drc {ignore_freq_hz false} [ipx::current_core]
set_property vitis_drc {ctrl_protocol ap_ctrl_hs} [ipx::current_core]
set_property supported_families { } [ipx::current_core]
set_property auto_family_support_level level_2 [ipx::current_core]

# Packaging Vivado IP
ipx::update_source_project_archive -component [ipx::current_core]
ipx::save_core [ipx::current_core]

package_xo -force -xo_path ${xoname} -kernel_name ${krnl_name} -ip_directory ${path_to_packaged} -kernel_files dummy_kernel.cpp
