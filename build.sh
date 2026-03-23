#!/bin/bash

# Build script for LSTM Inference C++ Library
# This script automates the build process across different platforms

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build"
INSTALL_PREFIX=""
USE_CUDA=OFF
NUM_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --cuda)
            USE_CUDA=ON
            shift
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        --clean)
            print_info "Cleaning build directory..."
            rm -rf "$BUILD_DIR"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --cuda               Enable CUDA support"
            echo "  --install-prefix DIR Set installation prefix"
            echo "  --jobs N             Number of parallel jobs (default: $NUM_JOBS)"
            echo "  --clean              Clean build directory before building"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_info "=== LSTM Inference C++ Build Script ==="
print_info "Build type: $BUILD_TYPE"
print_info "CUDA support: $USE_CUDA"
print_info "Parallel jobs: $NUM_JOBS"

# Check for ONNX Runtime
if [ -z "$ONNXRUNTIME_ROOT" ]; then
    print_warning "ONNXRUNTIME_ROOT not set"
    print_info "Searching for ONNX Runtime in standard locations..."
    
    # Try to find ONNX Runtime
    for dir in /usr/local/onnxruntime* /opt/onnxruntime* ~/onnxruntime* ./onnxruntime* ./external/onnxruntime; do
        if [ -d "$dir" ]; then
            export ONNXRUNTIME_ROOT="$dir"
            print_info "Found ONNX Runtime at: $ONNXRUNTIME_ROOT"
            break
        fi
    done
    
    if [ -z "$ONNXRUNTIME_ROOT" ]; then
        print_error "ONNX Runtime not found!"
        echo ""
        echo "Please either:"
        echo "  1. Set ONNXRUNTIME_ROOT environment variable"
        echo "  2. Download ONNX Runtime from: https://github.com/microsoft/onnxruntime/releases"
        echo ""
        echo "Example:"
        echo "  export ONNXRUNTIME_ROOT=/path/to/onnxruntime"
        exit 1
    fi
else
    print_info "Using ONNX Runtime from: $ONNXRUNTIME_ROOT"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
print_info "Configuring CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DUSE_CUDA="$USE_CUDA"
    -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT"
)

if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX")
fi

cmake .. "${CMAKE_ARGS[@]}"

# Build
print_info "Building..."
cmake --build . --config "$BUILD_TYPE" -j "$NUM_JOBS"

print_info "✓ Build completed successfully!"
echo ""
print_info "Build artifacts in: $BUILD_DIR"
print_info "Examples in: $BUILD_DIR/examples/"
echo ""
print_info "To install, run:"
echo "  cd $BUILD_DIR && sudo cmake --install ."
echo ""
print_info "To run examples:"
echo "  $BUILD_DIR/examples/physics_model_example /path/to/model.onnx"
