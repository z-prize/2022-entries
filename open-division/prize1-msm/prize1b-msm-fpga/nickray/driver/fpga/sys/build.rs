use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = cc::Build::new();

    let builder = builder
        .file("c/appxfer/appxfer.c")
        .file("c/appxfer/appxfer_f1/appxfer_f1.c")
        .file("c/upstream/fpga_libs/fpga_mgmt/fpga_mgmt.c")
        .file("c/upstream/fpga_libs/fpga_mgmt/fpga_mgmt_cmd.c")
        .file("c/upstream/fpga_libs/fpga_mgmt/fpga_hal_mbox.c")
        .file("c/upstream/fpga_libs/fpga_pci/fpga_pci.c")
        .file("c/upstream/fpga_libs/fpga_pci/fpga_pci_sysfs.c")
        .file("c/upstream/utils/io.c")
        .include("c/appxfer")
        .include("c/upstream/include")
        .include("c/upstream/fpga_libs/fpga_mgmt")
        .opt_level(3)
        .flag("-mavx2")
        // can't really fix upstream
        .warnings(false);

    builder.compile("fpga-sys");

    let bindings = bindgen::Builder::default()
        .header("c/appxfer/appxfer.h")
        .use_core()
        .ctypes_prefix("cty")
        .rustfmt_bindings(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    Ok(())
}
