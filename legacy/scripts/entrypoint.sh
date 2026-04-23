#!/bin/sh
# entrypoint.sh — prepend /opt/rope/gpu-libs to LD_LIBRARY_PATH if it is
# non-empty, then exec rope_demo.
#
# This lets the GPU backend be selected purely at runtime:
#   docker run -v /host/ort-cuda-libs:/opt/rope/gpu-libs:ro ...   → CUDA
#   docker run -v /host/ort-rocm-libs:/opt/rope/gpu-libs:ro ...   → ROCm
#   docker run ...                                                 → CPU

GPU_LIBS=/opt/rope/gpu-libs

if [ -d "$GPU_LIBS" ] && [ "$(ls -A "$GPU_LIBS" 2>/dev/null)" ]; then
    export LD_LIBRARY_PATH="${GPU_LIBS}:${LD_LIBRARY_PATH:-}"
fi

exec rope_demo "$@"
