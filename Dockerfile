# docker build -t zprize22-msm-gpu:latest .

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

RUN apt-get update && \
    # isntal auxiliary tools
    apt-get install -y curl && \
    # install clang
    apt-get -y install clang && \
    # install rust
    curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    # cleanup Docker
    rm -rf /var/lib/apt/lists/*

ENV NVCC_PREPEND_FLAGS="-ccbin=gcc -std=c++17" \
    PATH="/root/.cargo/bin:${PATH}"

CMD ["bash"]
