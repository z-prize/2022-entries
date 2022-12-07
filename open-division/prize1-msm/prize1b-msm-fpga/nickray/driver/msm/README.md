# msm-fpga

Assuming the FPGA image is loaded.

Default SIZE=26.

- `make point SIZE=<SIZE>` generates files `size<SIZE>.beta` and `size<SIZE>.points`.
- `make column SIZE=<SIZE>` calculates a column MSM using these points.
- `make full SIZE=<SIZE>` calculates a full (16 column) MSM using these points.

#### License

<sup>
Licensed under either of <a href="../LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="../LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>

