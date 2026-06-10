#!/usr/bin/env bash

jobs=("linux" "linux-cuda12" "mac-arm64" "windows" "windows-cuda12")
job_file=".github/workflows/build.yml"
display_str="⭐|✅|❌|node|Cache|🐳 docker (pull|exec)"

for i in {0..2}; do
    act -j "build-${jobs[i]}" -W $job_file --artifact-server-path .artifacts 2>&1 | grep -E "${display_str}"
done