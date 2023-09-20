# docker build -t zprize22-msm-gpu:latest .

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV NVCC_PREPEND_FLAGS="-ccbin=gcc -std=c++17"

RUN apt-get update && \
    # isntal auxiliary tools
    apt-get install -y curl && \
    # install clang
    apt-get -y install clang && \
    # install rust
    curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rm -rf /var/lib/apt/lists/*

CMD ["bash"]
