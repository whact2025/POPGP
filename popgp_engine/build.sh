#!/bin/bash
# POPGP Engine Build Script (Linux/macOS)

set -e # Exit on error

CONFIG="Release"
RUN_TESTS=false
CLEAN=false
CUDA_ARCH="native"

# Parse Args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) CONFIG="Debug" ;;
        --test) RUN_TESTS=true ;;
        --clean) CLEAN=true ;;
        --cuda-arch)
            shift
            if [[ "$#" -eq 0 ]]; then
                echo "Error: --cuda-arch requires a CMake CUDA architecture value."
                exit 1
            fi
            CUDA_ARCH="$1"
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cmake -DPOPGP_CUDA_ARCHITECTURE="$CUDA_ARCH" \
    -P cmake/ValidateCudaArchitecture.cmake

echo "--- POPGP Engine Build ($CONFIG, CUDA architecture $CUDA_ARCH) ---"

# 1. Clean
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# 2. Configure. Always rerun so requested architecture/configuration changes
# cannot be silently masked by an existing CMake cache.
echo "Configuring CMake..."

TOOLCHAIN=""
if [ -n "$VCPKG_ROOT" ]; then
    TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
else
    echo "Warning: VCPKG_ROOT not set."
fi

cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE="$CONFIG" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DPOPGP_REQUIRE_VISIBLE_CUDA_ARCH=ON $TOOLCHAIN

# 3. Build
echo "Building..."
cmake --build build --config $CONFIG

# 4. Tests
if [ "$RUN_TESTS" = true ]; then
    echo "Running Tests..."
    cmake -DPOPGP_BUILD_DIR="$PWD/build" -DPOPGP_CONFIG="$CONFIG" \
        -P cmake/VerifyCTestCount.cmake
    ctest --test-dir build -C "$CONFIG" --output-on-failure --no-tests=error
fi

echo "Build Complete!"
