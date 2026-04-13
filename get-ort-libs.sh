#!/usr/bin/env bash
# get-ort-libs.sh — download prebuilt ORT shared libraries for GPU backends.
#
# Produces a directory that can be mounted into the container:
#   docker run -v "$(pwd)/ort-libs/cuda":/opt/rope/gpu-libs:ro --gpus all rope:latest
#   docker run -v "$(pwd)/ort-libs/rocm":/opt/rope/gpu-libs:ro rope:latest
#
# Usage:
#   ./get-ort-libs.sh cuda [ORT_VERSION]   # CUDA 12 prebuilt from GitHub
#   ./get-ort-libs.sh rocm [ORT_VERSION]   # ROCm build from PyPI wheel
#   ./get-ort-libs.sh all  [ORT_VERSION]   # both
#
# ORT_VERSION defaults to the version in external/onnxruntime/VERSION_NUMBER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${1:-all}"

# Resolve ORT version from submodule if not given.
if [[ -n "${2:-}" ]]; then
    ORT_VERSION="$2"
elif [[ -f "${SCRIPT_DIR}/external/onnxruntime/VERSION_NUMBER" ]]; then
    ORT_VERSION="$(cat "${SCRIPT_DIR}/external/onnxruntime/VERSION_NUMBER")"
else
    ORT_VERSION="1.20.1"
fi

OUT_DIR="${SCRIPT_DIR}/ort-libs"
SKIP_PATTERN='libcustom_op_|libtest_|libexample_|libonnxruntime_runtime_path_'

echo "ORT version : ${ORT_VERSION}"
echo "Output root : ${OUT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# CUDA — official prebuilt from GitHub Releases
# ---------------------------------------------------------------------------
get_cuda() {
    local dest="${OUT_DIR}/cuda"
    mkdir -p "$dest"

    echo "==> Downloading ORT CUDA prebuilt..."
    wget -qO /tmp/ort-cuda.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"

    local tmp
    tmp="$(mktemp -d)"
    tar -xzf /tmp/ort-cuda.tgz -C "$tmp" --strip-components=1
    rm /tmp/ort-cuda.tgz

    # Copy only the shared libraries.
    find "${tmp}/lib" -maxdepth 1 -name '*.so*' \
        | grep -Ev "${SKIP_PATTERN}" \
        | xargs -I{} cp -a {} "$dest/"

    rm -rf "$tmp"
    echo "   → ${dest}  ($(ls "$dest" | wc -l) files, $(du -sh "$dest" | cut -f1))"
}

# ---------------------------------------------------------------------------
# ROCm — extract .so files from the onnxruntime-rocm PyPI wheel
#
#  The wheel is a zip archive.  The shared libraries live in:
#    onnxruntime/capi/libonnxruntime*.so*
# ---------------------------------------------------------------------------
get_rocm() {
    local dest="${OUT_DIR}/rocm"
    mkdir -p "$dest"

    # Requires pip or pip3 in PATH.
    if ! command -v pip3 &>/dev/null && ! command -v pip &>/dev/null; then
        echo "Error: pip/pip3 not found.  Install Python + pip to download ROCm ORT."
        exit 1
    fi
    PIP="${PIP:-pip3}"

    echo "==> Downloading onnxruntime-rocm==${ORT_VERSION} wheel..."
    local whl_dir
    whl_dir="$(mktemp -d)"
    # --no-deps: we only want the ORT wheel, not its Python dependencies.
    $PIP download --no-deps --no-build-isolation \
        "onnxruntime-rocm==${ORT_VERSION}" \
        -d "$whl_dir" 2>/dev/null \
    || {
        echo ""
        echo "  Note: onnxruntime-rocm==${ORT_VERSION} not found on PyPI."
        echo "  Try a nearby version, e.g.:"
        echo "    ./get-ort-libs.sh rocm 1.20.0"
        echo "  Or build ORT from source for ROCm."
        rm -rf "$whl_dir"
        return 1
    }

    local whl
    whl="$(find "$whl_dir" -name 'onnxruntime_rocm-*.whl' | head -1)"
    if [[ -z "$whl" ]]; then
        echo "Error: wheel not found after download."
        rm -rf "$whl_dir"
        return 1
    fi

    echo "   Extracting from $(basename "$whl")..."
    local unpack_dir
    unpack_dir="$(mktemp -d)"
    unzip -q "$whl" -d "$unpack_dir"

    # The .so files sit in onnxruntime/capi/ inside the wheel.
    find "${unpack_dir}" -name 'libonnxruntime*.so*' \
        | xargs -I{} cp -a {} "$dest/"

    rm -rf "$whl_dir" "$unpack_dir"
    echo "   → ${dest}  ($(ls "$dest" | wc -l) files, $(du -sh "$dest" | cut -f1))"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$BACKEND" in
    cuda) get_cuda ;;
    rocm) get_rocm ;;
    all)  get_cuda; echo ""; get_rocm ;;
    *)
        echo "Usage: $0 cuda|rocm|all [ORT_VERSION]"
        exit 1
        ;;
esac

echo ""
echo "==> Done.  Mount in the container with:"
echo "    CUDA:   -v \"${OUT_DIR}/cuda\":/opt/rope/gpu-libs:ro  --gpus all"
echo "    ROCm:   -v \"${OUT_DIR}/rocm\":/opt/rope/gpu-libs:ro  --device /dev/kfd --device /dev/dri"
