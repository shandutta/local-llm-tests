# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder
ARG LLAMA_CPP_REPO=https://github.com/ggerganov/llama.cpp.git
ARG LLAMA_CPP_REF=master
WORKDIR /opt
RUN apt-get update && \
    apt-get install -y --no-install-recommends git cmake build-essential python3 libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so && \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
RUN git clone --depth=1 --branch "$LLAMA_CPP_REF" "$LLAMA_CPP_REPO" llama.cpp
WORKDIR /opt/llama.cpp
RUN cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && \
    cmake --build build -j$(nproc)
RUN cmake --install build --prefix /opt/llama-install

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
WORKDIR /workspace
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 ca-certificates libcurl4 libgomp1 && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/llama-install /opt/llama
ENV PATH="/opt/llama/bin:${PATH}"
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENTRYPOINT ["/entrypoint.sh"]
