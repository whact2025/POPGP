cmake_minimum_required(VERSION 3.25)

# Build for the Native GPU Architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

project(popgp_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

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

