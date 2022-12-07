# driver

Driver for ZPrize FPGA MSM submission

[algebra](algebra) and [bls12-377](bls12-377) are forks of [arkworks](https://github.com/arkworks-rs),
containing code to convert BLS12-377 into preprocessed Twisted Edwards coordinates, and back.

[fpga](fpga) contains Rust bindings to use AWS F1 instances using a custom protocol.

[msm](msm) contains code to calculate large-size MSM instances, using the [FPGA app](../cl).

#### License

<sup>
Licensed under either of <a href="LICENSE-APACHE">Apache License, Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.
</sup>

<br>

<sub>
Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
</sub>

[arkworks]: https://github.com/arkworks-rs
