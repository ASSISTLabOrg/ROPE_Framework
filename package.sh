#!/usr/bin/env bash
# package.sh — collect rope_demo and its non-system shared libraries into
# dist/ so that Dockerfile.prebuilt can build a container without compiling
# anything inside Docker.
#
# Usage:
#   ./package.sh                  # uses build/rope_demo (default)
#   ./package.sh path/to/binary   # explicit binary path
#
# Output layout:
#   dist/
#     rope_demo          — the stripped executable
#     libs/              — all non-system .so files (with versioned names)
#
# After running this script:
#   docker build -f Dockerfile.prebuilt -t rope:prebuilt .

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${1:-${SCRIPT_DIR}/build/rope_demo}"
DIST_DIR="${SCRIPT_DIR}/dist"

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ ! -x "${BINARY}" ]]; then
    echo "Error: binary not found or not executable: ${BINARY}"
    echo "Run 'cmake --build build' first, then re-run package.sh."
    exit 1
fi

BINARY="$(realpath "${BINARY}")"

echo "==> Packaging ${BINARY}"
echo ""

# ---------------------------------------------------------------------------
# Collect non-system shared libraries
# ---------------------------------------------------------------------------
# "System" paths ship with the base ubuntu:noble image — we leave those out.
SYSTEM_PREFIXES=("/lib/" "/lib64/" "/usr/lib/" "/usr/lib64/" "/usr/local/lib/")

is_system_lib() {
    local path="$1"
    for prefix in "${SYSTEM_PREFIXES[@]}"; do
        [[ "$path" == "${prefix}"* ]] && return 0
    done
    return 1
}

# Track which source directories to harvest (deduplicated).
declare -A harvest_dirs

while IFS= read -r line; do
    # ldd format: "\tname => /resolved/path (0xADDR)"  or  "\t/abs/path (0xADDR)"
    if [[ "$line" =~ '=>'[[:space:]]+([^[:space:]]+)[[:space:]]+\(0x ]]; then
        path="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]+(/[^[:space:]]+)[[:space:]]+\(0x ]]; then
        path="${BASH_REMATCH[1]}"
    else
        continue
    fi

    [[ "$path" == "not" ]] && continue          # "not found" line
    is_system_lib "$path"  && continue          # system library

    dir="$(dirname "$(realpath "$path")")"
    harvest_dirs["$dir"]=1
done < <(ldd "${BINARY}" 2>/dev/null)

# ---------------------------------------------------------------------------
# Also harvest the directory of libonnxruntime itself — it contains provider
# SOs that ORT loads via dlopen() at runtime and won't appear in ldd output.
# ---------------------------------------------------------------------------
ORT_PATH="$(ldd "${BINARY}" 2>/dev/null | awk '/libonnxruntime\.so/{print $3; exit}')"
if [[ -n "${ORT_PATH:-}" ]]; then
    ORT_DIR="$(dirname "$(realpath "${ORT_PATH}")")"
    harvest_dirs["${ORT_DIR}"]=1
fi

# ---------------------------------------------------------------------------
# Copy libraries
# ---------------------------------------------------------------------------
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/libs"

# Patterns to skip: ORT test/example plugins not needed at runtime.
SKIP_PATTERN='libcustom_op_|libtest_|libexample_|libonnxruntime_runtime_path_|libjitbackend_test|libtorchbind_test|libtorch_python|libnnapi_backend'

for dir in "${!harvest_dirs[@]}"; do
    echo "   harvesting ${dir}/"
    while IFS= read -r -d '' so; do
        name="$(basename "$so")"
        if echo "$name" | grep -qE "${SKIP_PATTERN}"; then
            continue
        fi
        # -a: preserve symlinks; -L would dereference — we want the symlink chain.
        cp -a "$so" "${DIST_DIR}/libs/" 2>/dev/null || true
    done < <(find "$dir" -maxdepth 1 -name '*.so*' -print0)
done

# ---------------------------------------------------------------------------
# Copy and strip the binary
# ---------------------------------------------------------------------------
cp "${BINARY}" "${DIST_DIR}/rope_demo"
# Strip debug symbols to reduce image size (comment out to keep them).
strip --strip-unneeded "${DIST_DIR}/rope_demo" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==> dist/ contents:"
printf "    %-40s %s\n" "rope_demo" "$(du -sh "${DIST_DIR}/rope_demo" | cut -f1)"
printf "    %-40s %s\n" "libs/ ($(ls "${DIST_DIR}/libs/" | wc -l) files)" \
    "$(du -sh "${DIST_DIR}/libs" | cut -f1)"
echo ""
echo "==> Next steps:"
echo "    CPU only:"
echo "      docker build -f Dockerfile.prebuilt -t rope:prebuilt ."
echo ""
echo "    NVIDIA / CUDA:"
echo "      docker build -f Dockerfile.prebuilt -t rope:prebuilt-cuda \\"
echo "        --build-arg GPU_BACKEND=cuda ."
echo ""
echo "    AMD / ROCm:"
echo "      docker build -f Dockerfile.prebuilt -t rope:prebuilt-rocm \\"
echo "        --build-arg GPU_BACKEND=rocm ."
