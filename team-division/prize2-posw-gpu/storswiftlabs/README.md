# ZPrize PoSW on GPU

## Build

```
rustup install stable
```
After that, use cargo, the standard Rust build tool, to build the libraries:

```
git clone https://github.com/storswiftlabs/zprize-gpu-posw.git
cd snarkVM/msm
cargo build --release --features cuda && cd ..
```
## Bench
- Hardware: AMD EPYC 7402P 24-Core Processor 16cores; 1 NVIDIA RTX A4000 Graphics Card
- Run: export MAX_FFT_LEN=130000 && export FFT_LEVEL=0 && export CLEVEL=3 && export MAX_SCALARS_LEN=4 && export WORK_COUNT=4,12,3,3,16,8,12,12,32,10 && export ENABLE_PROVE_LOG=1 && ./target/release/hashrate -g 0 -j 16 -t 2 -l 800 -s
- Performance result:293P/20s


