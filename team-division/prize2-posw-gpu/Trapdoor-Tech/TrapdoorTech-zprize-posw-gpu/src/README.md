# ZPrize test harnesses

This repository contains test harnesses for the PoSW challenge and the Marlin verifier challenge.

## Setup

To test your competition entry, it is important that you first import it into the harness in place of the default `snarkVM` import. To do this, simply open the `Cargo.toml` file in the repository root and replace the dependencies to point towards your own `snarkVM` fork.

Don't forget to enable the desired feature flags, if you're running a non-standard algorithm! (GPU for instance)

## Usage

The binary supports two testing modes, one for each challenge.

To run the PoSW test, run:

```bash
cargo run --release -- proving
```

To run the Marlin verifier test, run:

```bash
cargo run --release -- verifying
```
