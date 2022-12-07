## Table of Contents

* [1. PoSW harness](#1-overview)
* [2. The concurrent implementation](#2-build-guide)
* [3. best performing machine configuration](#3-usage-guide)

## 1. PoSW harness
### 1.1 location of harness
```bash
  dpc/examples/provings.rs
```

### 1.2 how to build
```bash
  cargo build --release --example proving --manifest-path ./dpc/Cargo.toml
```

### 1.3 run the release binary
```bash
  ./target/release/examples/proving
```

## 2. document for optimizations
```bash
  ./optimizations_for_posw.pdf
```


### 3 Our best performing machine configuration
| platform | GPU Cards         |          Cpu       |    Our Test results |  total cost per hour | System Version | 
|---------|:-----------------:|--------------------|---------------------|----------------------|----------------------|
| corewave cloud    |    Quadro RTX 4000 x 3| AMD EPYC 7402P 30-Core Processor|   about 500 proofs |   $1.17 | Ubuntu 22.04.1 LTS | 

