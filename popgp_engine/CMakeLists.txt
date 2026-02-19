cmake_minimum_required(VERSION 3.25)

# Build for the Native GPU Architecture (Fixes silent launch failures)
# Must be set BEFORE project()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Force Ampere (sm_86) for RTX A6000
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

project(popgp_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Dependencies (via vcpkg in root)
find_package(fmt CONFIG REQUIRED)

# Include Directories (Global)
include_directories(kernel/include)

# Enable CTest at Root
enable_testing()

# Components
add_subdirectory(kernel)
# add_subdirectory(renderer) # Uncomment when renderer is ready
