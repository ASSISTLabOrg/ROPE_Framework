#!/usr/bin/env bash
# DemoCpp.sh — run the same forecast and interpolations as DemoPy.py
# using the C++ ROPE REPL.
#
# Pipe a sequence of commands into rope_demo via stdin, then exit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/build/rope_demo" "${SCRIPT_DIR}/config/rope.conf" <<'EOF'
run 2024-02-09 00:00:00 120
query hold   2024-02-10 00:30:00 10.5 25.0 400.0
query interp 2024-02-10 00:30:00 10.5 25.0 400.0
quit
EOF
