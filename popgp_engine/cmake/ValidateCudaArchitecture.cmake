if(NOT DEFINED POPGP_CUDA_ARCHITECTURE OR POPGP_CUDA_ARCHITECTURE STREQUAL "")
    message(FATAL_ERROR "POPGP_CUDA_ARCHITECTURE is required")
endif()

if(NOT POPGP_CUDA_ARCHITECTURE MATCHES
        "^(native|all|all-major|[0-9]+(-real|-virtual)?)$")
    message(FATAL_ERROR
        "Invalid CUDA architecture '${POPGP_CUDA_ARCHITECTURE}'. Expected native, "
        "all, all-major, or a numeric nn/nn-real/nn-virtual value."
    )
endif()
