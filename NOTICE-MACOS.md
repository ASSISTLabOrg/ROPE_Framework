## Notice - MacOS Users Only

The current release has a small issue with path resolution. This will be fixed within 24 hours in the next version, but until then the following may be useful.

Command to run the fix:

INSTALL_DIR="/path/to/rope_framework-0.3.6-macos-arm64-cpu"

install_name_tool \
    -add_rpath "@executable_path/../lib" \
    -add_rpath "@loader_path/../lib" \
    -add_rpath "${INSTALL_DIR}/lib" \
    "${INSTALL_DIR}/bin/rope"

Requires Xcode Command Line Tools (xcode-select --install).