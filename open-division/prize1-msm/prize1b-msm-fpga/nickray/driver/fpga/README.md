# fpga

Rust bindings to interact with AWS F1 FPGAs.

Uses an abstraction implemented in `c/appxfer`.

`fpga` contains the actual bindings.

`fpga-sys` (in `sys/`) consists of `bindgen` bindings of the `fpga_libs`
from the directory `sdk/userspace` of the [aws/aws-fpga][aws-fpga] repo,
which are included in `sys/c/upstream` to avoid git submodules.

Disable the default feature to build without C bindings; this offers
an `fpga::Null` device.

[aws-fpga]: https://github.com/aws/aws-fpga/tree/master/sdk/userspace/

#### License

<sup>
`aws-fpga` is licensed under Apache 2.0.
These bindings are licensed under either of <a href="../LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="../LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>

