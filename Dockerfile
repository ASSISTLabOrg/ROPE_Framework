# syntax=docker/dockerfile:1.7
# =============================================================================
#  ROPE — container build
#  Base: ubuntu:noble (Ubuntu 24.04 LTS)
#
#  All external libraries (ORT, LibTorch) are downloaded as prebuilt binaries.
#  No external source code is compiled inside Docker.
#
#  Build args
#  ----------
#  ORT_VERSION      1.20.1  — ORT release tag (github.com/microsoft/onnxruntime)
#                             Match this to external/onnxruntime/VERSION_NUMBER.
#  TORCH_VERSION    2.5.1   — LibTorch release
#  ROPE_USE_LIBTORCH  OFF   — set ON to include the TorchScript backend (+~2 GB)
#  NPROC            (blank) — parallel compile jobs for ROPE itself; defaults to nproc
#
#  Build:
#    docker build -t rope:latest .
#    docker build -t rope:latest --build-arg ROPE_USE_LIBTORCH=ON .
#
#  Run — CPU:
#    docker run --rm -it \
#      -v /path/to/exported:/app/exported:ro \
#      -v /path/to/data:/app/data:ro \
#      rope:latest
#
#  Run — NVIDIA (requires nvidia-container-toolkit on host):
#    docker run --rm -it --gpus all \
#      -v /path/to/ort-cuda-libs:/opt/rope/gpu-libs:ro \
#      -v /path/to/exported:/app/exported:ro \
#      -v /path/to/data:/app/data:ro \
#      rope:latest
#
#  Run — AMD (requires amdgpu driver on host):
#    docker run --rm -it \
#      --device /dev/kfd --device /dev/dri --group-add video \
#      -v /path/to/ort-rocm-libs:/opt/rope/gpu-libs:ro \
#      -v /path/to/exported:/app/exported:ro \
#      -v /path/to/data:/app/data:ro \
#      rope:latest
#
#  GPU libs are loaded at runtime via LD_LIBRARY_PATH — no image rebuild needed
#  to switch backends.  See docs/gpu-libs.md for how to prepare each backend.
# =============================================================================

ARG ORT_VERSION=1.20.1
ARG TORCH_VERSION=2.5.1
ARG ROPE_USE_LIBTORCH=OFF

# =============================================================================
# Stage 1 — builder
# =============================================================================
FROM ubuntu:noble AS builder

ARG ORT_VERSION
ARG TORCH_VERSION
ARG ROPE_USE_LIBTORCH
ARG NPROC

# ---------------------------------------------------------------------------
# 1a. Build tools
# ---------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        wget \
        unzip \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 1b. ONNX Runtime — CPU prebuilt (provides headers + libonnxruntime.so)
#
#  The CPU prebuilt is used for compilation regardless of the runtime GPU
#  backend.  ORT's shared library ABI is stable — the CPU, CUDA, and ROCm
#  builds are API-compatible.  Swapping the .so at runtime is safe.
# ---------------------------------------------------------------------------
RUN wget -qO /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" && \
    mkdir -p /opt/onnxruntime && \
    tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 && \
    rm /tmp/ort.tgz

# ---------------------------------------------------------------------------
# 1c. LibTorch — optional (~2 GB download; set ROPE_USE_LIBTORCH=ON to enable)
# ---------------------------------------------------------------------------
RUN if [ "$ROPE_USE_LIBTORCH" = "ON" ]; then \
      URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcpu.zip" && \
      wget -qO /tmp/libtorch.zip "$URL" && \
      unzip -q /tmp/libtorch.zip -d /opt && \
      mv /opt/libtorch /opt/libtorch-install && \
      rm /tmp/libtorch.zip ; \
    else \
      mkdir -p /opt/libtorch-install/lib ; \
    fi

# ---------------------------------------------------------------------------
# 1d. Build ROPE C++
# ---------------------------------------------------------------------------
WORKDIR /rope

COPY CMakeLists.txt .
COPY src/            src/
COPY include/        include/

RUN JOBS="${NPROC:-$(nproc)}" && \
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/opt/onnxruntime \
      -DROPE_USE_LIBTORCH="${ROPE_USE_LIBTORCH}" \
      -DLIBTORCH_ROOT=/opt/libtorch-install \
      -DROPE_USE_DNNL=OFF \
    && cmake --build build --parallel "$JOBS"

# =============================================================================
# Stage 2 — runtime
# =============================================================================
FROM ubuntu:noble AS runtime

LABEL org.opencontainers.image.title="ROPE Thermospheric Density Forecast" \
      org.opencontainers.image.description="C++ ROPE runtime — GPU backend selectable at runtime via volume mount"

# ---------------------------------------------------------------------------
# 2a. Runtime OS deps (minimal — no GPU packages baked in)
# ---------------------------------------------------------------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2b. ORT + LibTorch shared libraries and binary
# ---------------------------------------------------------------------------
COPY --from=builder /opt/onnxruntime/lib/      /opt/rope/libs/
COPY --from=builder /opt/libtorch-install/lib/ /opt/rope/libs/
COPY --from=builder /rope/build/rope_demo      /usr/local/bin/rope_demo

RUN echo /opt/rope/libs > /etc/ld.so.conf.d/rope.conf && ldconfig

# ---------------------------------------------------------------------------
# 2c. Application data
# ---------------------------------------------------------------------------
WORKDIR /app

COPY config/   ./config/
COPY exported/ ./exported/
COPY data/     ./data/

VOLUME ["/app/exported", "/app/data"]

# /opt/rope/gpu-libs is the runtime GPU swap point.
# Mount a directory containing libonnxruntime.so (CUDA or ROCm build) here
# and the entrypoint will prepend it to LD_LIBRARY_PATH automatically.
VOLUME ["/opt/rope/gpu-libs"]

# ---------------------------------------------------------------------------
# 2d. Entrypoint — handles optional GPU lib injection
# ---------------------------------------------------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV ORT_LOGGING_LEVEL=3

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/app/config/rope.conf"]
