cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED POPGP_CUDA_BINARY OR NOT EXISTS "${POPGP_CUDA_BINARY}")
    message(FATAL_ERROR "POPGP_CUDA_BINARY must name an existing binary")
endif()
if(NOT DEFINED POPGP_CUDA_ARCHITECTURE)
    message(FATAL_ERROR "POPGP_CUDA_ARCHITECTURE is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/ValidateCudaArchitecture.cmake")

if(NOT DEFINED POPGP_CUOBJDUMP_ELF_OUTPUT)
    if(NOT DEFINED POPGP_CUOBJDUMP_EXECUTABLE OR
            NOT EXISTS "${POPGP_CUOBJDUMP_EXECUTABLE}")
        find_program(POPGP_CUOBJDUMP_EXECUTABLE NAMES cuobjdump REQUIRED)
    endif()
    execute_process(
        COMMAND "${POPGP_CUOBJDUMP_EXECUTABLE}" --list-elf "${POPGP_CUDA_BINARY}"
        RESULT_VARIABLE _elf_result
        OUTPUT_VARIABLE POPGP_CUOBJDUMP_ELF_OUTPUT
        ERROR_VARIABLE _elf_error
        ENCODING UTF-8
    )
    if(NOT _elf_result EQUAL 0)
        message(FATAL_ERROR "cuobjdump --list-elf failed: ${_elf_error}")
    endif()
endif()
if(NOT DEFINED POPGP_CUOBJDUMP_PTX_OUTPUT)
    execute_process(
        COMMAND "${POPGP_CUOBJDUMP_EXECUTABLE}" --list-ptx "${POPGP_CUDA_BINARY}"
        RESULT_VARIABLE _ptx_result
        OUTPUT_VARIABLE POPGP_CUOBJDUMP_PTX_OUTPUT
        ERROR_VARIABLE _ptx_error
        ENCODING UTF-8
    )
    if(NOT _ptx_result EQUAL 0)
        message(FATAL_ERROR "cuobjdump --list-ptx failed: ${_ptx_error}")
    endif()
endif()

string(REGEX MATCHALL "sm_[0-9]+" _elf_tokens "${POPGP_CUOBJDUMP_ELF_OUTPUT}")
string(REGEX MATCHALL "sm_[0-9]+" _ptx_tokens "${POPGP_CUOBJDUMP_PTX_OUTPUT}")
list(REMOVE_DUPLICATES _elf_tokens)
list(REMOVE_DUPLICATES _ptx_tokens)

set(_visible_arches "")
if(DEFINED POPGP_NATIVE_ARCHITECTURES)
    set(_visible_arches ${POPGP_NATIVE_ARCHITECTURES})
elseif(POPGP_REQUIRE_VISIBLE_CUDA_ARCH)
    find_program(_nvidia_smi NAMES nvidia-smi REQUIRED)
    execute_process(
        COMMAND "${_nvidia_smi}" --query-gpu=compute_cap --format=csv,noheader
        RESULT_VARIABLE _smi_result
        OUTPUT_VARIABLE _smi_output
        ERROR_VARIABLE _smi_error
        ENCODING UTF-8
    )
    if(NOT _smi_result EQUAL 0)
        message(FATAL_ERROR "nvidia-smi compute-capability query failed: ${_smi_error}")
    endif()
    string(REPLACE "\r" "" _smi_output "${_smi_output}")
    string(REPLACE "\n" ";" _smi_lines "${_smi_output}")
    foreach(_cap IN LISTS _smi_lines)
        string(STRIP "${_cap}" _cap)
        if(NOT _cap STREQUAL "")
            string(REPLACE "." "" _cap "${_cap}")
            list(APPEND _visible_arches "${_cap}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _visible_arches)
endif()

set(_requested "${POPGP_CUDA_ARCHITECTURE}")
if(_requested MATCHES "^([0-9]+)(-real|-virtual)?$")
    set(_requested_numeric "${CMAKE_MATCH_1}")
    set(_requested_kind "${CMAKE_MATCH_2}")
    if(_requested_kind STREQUAL "-virtual")
        set(_tokens ${_ptx_tokens})
        set(_kind "PTX")
    else()
        set(_tokens ${_elf_tokens})
        set(_kind "ELF cubin")
    endif()
    if(NOT _tokens)
        message(FATAL_ERROR "No ${_kind} code was found for requested architecture ${_requested}")
    endif()
    foreach(_token IN LISTS _tokens)
        if(NOT _token STREQUAL "sm_${_requested_numeric}")
            message(FATAL_ERROR
                "Binary contains ${_token}, not requested sm_${_requested_numeric}: "
                "${POPGP_CUDA_BINARY}"
            )
        endif()
    endforeach()
    if(POPGP_REQUIRE_VISIBLE_CUDA_ARCH AND _visible_arches AND
            NOT _requested_numeric IN_LIST _visible_arches)
        message(FATAL_ERROR
            "Requested sm_${_requested_numeric} does not match a visible GPU architecture "
            "(${_visible_arches})"
        )
    endif()
elseif(_requested STREQUAL "native")
    if(NOT _visible_arches)
        message(FATAL_ERROR "The native architecture cannot be verified without a visible GPU")
    endif()
    if(NOT _elf_tokens)
        message(FATAL_ERROR "The native build contains no embedded ELF cubins")
    endif()
    foreach(_visible IN LISTS _visible_arches)
        if(NOT "sm_${_visible}" IN_LIST _elf_tokens)
            message(FATAL_ERROR
                "Native binary lacks code for visible sm_${_visible}: ${_elf_tokens}"
            )
        endif()
    endforeach()
else()
    if(NOT _elf_tokens AND NOT _ptx_tokens)
        message(FATAL_ERROR "CUDA binary contains neither ELF cubins nor PTX")
    endif()
    if(POPGP_REQUIRE_VISIBLE_CUDA_ARCH)
        foreach(_visible IN LISTS _visible_arches)
            if(NOT "sm_${_visible}" IN_LIST _elf_tokens AND
                    NOT "sm_${_visible}" IN_LIST _ptx_tokens)
                message(FATAL_ERROR
                    "CUDA binary lacks code for visible sm_${_visible}: "
                    "ELF=${_elf_tokens}; PTX=${_ptx_tokens}"
                )
            endif()
        endforeach()
    endif()
endif()

message(STATUS
    "Verified CUDA binary ${POPGP_CUDA_BINARY}: ELF=${_elf_tokens}; PTX=${_ptx_tokens}"
)
