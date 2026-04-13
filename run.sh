#!/usr/bin/env bash
# run.sh — build (if needed) and execute the ROPE demo.
#
# Usage:
#   ./run.sh                          # use default config/rope.conf
#   ./run.sh path/to/custom.conf      # use an alternative config file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/rope_demo"
CONFIG="${1:-${SCRIPT_DIR}/config/rope.conf}"

# ---------------------------------------------------------------------------
# Build if the binary is missing or sources are newer.
# ---------------------------------------------------------------------------
needs_build() {
    [[ ! -x "${BINARY}" ]] && return 0
    # Rebuild if any source or header is newer than the binary.
    while IFS= read -r -d '' f; do
        [[ "${f}" -nt "${BINARY}" ]] && return 0
    done < <(find "${SCRIPT_DIR}/src" "${SCRIPT_DIR}/include" \
                  -name '*.cpp' -o -name '*.hpp' -print0 2>/dev/null)
    return 1
}

if needs_build; then
    echo "==> Configuring and building rope_demo..."
    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build "${BUILD_DIR}" --parallel
    echo ""
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
exec "${BINARY}" "${CONFIG}"
