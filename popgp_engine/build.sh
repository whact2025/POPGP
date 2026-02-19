#!/bin/bash
# POPGP Engine Build Script (Linux/macOS)

set -e # Exit on error

CONFIG="Release"
RUN_TESTS=false
CLEAN=false

# Parse Args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --debug) CONFIG="Debug" ;;
        --test) RUN_TESTS=true ;;
        --clean) CLEAN=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "--- POPGP Engine Build ($CONFIG) ---"

# 1. Clean
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# 2. Configure
if [ ! -d "build" ]; then
    echo "Configuring CMake..."
    
    TOOLCHAIN=""
    if [ -n "$VCPKG_ROOT" ]; then
        TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
    else
        echo "Warning: VCPKG_ROOT not set."
    fi

    cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=$CONFIG $TOOLCHAIN
fi

# 3. Build
echo "Building..."
cmake --build build --config $CONFIG

# 4. Tests
if [ "$RUN_TESTS" = true ]; then
    echo "Running Tests..."
    cd build
    ctest -C $CONFIG --output-on-failure
    cd ..
fi

echo "Build Complete!"
