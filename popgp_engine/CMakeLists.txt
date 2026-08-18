cmake_minimum_required(VERSION 3.25)

# Build for the GPU visible at configure time unless the caller pins an
# architecture explicitly (for example, -DCMAKE_CUDA_ARCHITECTURES=120).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(popgp_engine LANGUAGES CXX)

# CUDA 12.8 is the first toolkit release that can emit Blackwell sm_120 code.
# Resolve and version-check the toolkit before enabling the CUDA language so an
# older compiler fails with a clear configure-time diagnostic rather than a late
# and opaque architecture error.
find_package(CUDAToolkit 12.8 REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 12.8)
    message(FATAL_ERROR
        "POPGP requires CUDA Toolkit 12.8 or newer; found ${CUDAToolkit_VERSION}."
    )
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH
        "CUDA compiler selected from the version-checked toolkit"
    )
endif()
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

option(
    POPGP_REQUIRE_VISIBLE_CUDA_ARCH
    "Require generated device code to include every locally visible GPU architecture"
    ON
)

# Dependencies (via vcpkg) -- header-only to avoid runtime DLL dependency
find_package(fmt CONFIG REQUIRED)

# Include Directories (Global)
include_directories(kernel/include)
include_directories(renderer/include)

# Enable CTest
enable_testing()

# Components
add_subdirectory(kernel)
add_subdirectory(renderer)

