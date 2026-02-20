
# Renderer Component (CUDA Regge Calculus & Embedding)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

add_library(popgp_render SHARED
    src/regge.cu
    src/embedding.cu
)

# Use CUDA Compiler
set_target_properties(popgp_render PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

target_link_libraries(popgp_render PRIVATE Eigen3::Eigen CUDA::cudart)

target_include_directories(popgp_render PUBLIC 
    include
)

# Export for Python
target_compile_definitions(popgp_render PRIVATE POPGP_RENDER_EXPORTS)

# MSVC Flags
if(MSVC)
    target_compile_options(popgp_render PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

