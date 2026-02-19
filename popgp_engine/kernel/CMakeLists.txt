# Kernel Component

# Find CUDA Toolkit for Runtime API (cudart)
find_package(CUDAToolkit REQUIRED)

# Kernel Library (Phase Flow) - Build as SHARED for Python/DLL usage
add_library(phase_flow SHARED
    src/phase_flow.cu
    src/area_law.cu
)

# Export symbols for Windows/Linux
target_compile_definitions(phase_flow PRIVATE POPGP_KERNEL_EXPORTS)

# Force UTF-8 for MSVC (Required by fmt 10+)
if(MSVC)
    target_compile_options(phase_flow PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

target_link_libraries(phase_flow PRIVATE fmt::fmt)
set_target_properties(phase_flow PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

# Post-Build: Copy DLL/SO to python directory for easy import
add_custom_command(TARGET phase_flow POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:phase_flow>
    ${CMAKE_CURRENT_SOURCE_DIR}/python/$<TARGET_FILE_NAME:phase_flow>
    COMMENT "Deploying shared library to Python artifact folder"
)

# Simulation Executable (Benchmark/Test Harness)
# Now a C++ file, linking against CUDA Runtime explicitly
add_executable(popgp_sim src/main.cpp)

if(MSVC)
    target_compile_options(popgp_sim PRIVATE /utf-8)
endif()

# Link against phase_flow, fmt, and CUDA Runtime
target_link_libraries(popgp_sim PRIVATE phase_flow fmt::fmt CUDA::cudart)

# Unit Tests
enable_testing()
find_package(GTest CONFIG REQUIRED)

add_executable(popgp_tests 
    tests/test_phase_flow.cu
    tests/test_area_law.cu
)
if(MSVC)
    target_compile_options(popgp_tests PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
target_link_libraries(popgp_tests PRIVATE phase_flow GTest::gtest GTest::gtest_main)

# Register Tests
include(GoogleTest)
gtest_discover_tests(popgp_tests)
