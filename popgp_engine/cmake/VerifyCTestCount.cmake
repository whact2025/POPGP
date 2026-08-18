if(NOT DEFINED POPGP_BUILD_DIR OR POPGP_BUILD_DIR STREQUAL "")
    message(FATAL_ERROR "POPGP_BUILD_DIR is required")
endif()

set(_expected_file "${POPGP_BUILD_DIR}/popgp_expected_native_tests.txt")
if(NOT EXISTS "${_expected_file}")
    message(FATAL_ERROR "Expected native-test count file is missing: ${_expected_file}")
endif()
file(READ "${_expected_file}" _expected)
string(STRIP "${_expected}" _expected)
if(NOT _expected MATCHES "^[0-9]+$" OR _expected STREQUAL "0")
    message(FATAL_ERROR "Invalid expected native-test count '${_expected}'")
endif()

find_program(_ctest_executable NAMES ctest REQUIRED)
execute_process(
    COMMAND "${_ctest_executable}" --test-dir "${POPGP_BUILD_DIR}" -C "${POPGP_CONFIG}" -N
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error
    ENCODING UTF-8
)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Could not enumerate native tests:\n${_output}${_error}")
endif()
if(NOT _output MATCHES "Total Tests: ([0-9]+)")
    message(FATAL_ERROR "CTest did not report a total test count:\n${_output}")
endif()
set(_actual "${CMAKE_MATCH_1}")
if(NOT _actual STREQUAL _expected)
    message(FATAL_ERROR
        "Expected ${_expected} native tests, but CTest discovered ${_actual}.\n${_output}"
    )
endif()
message(STATUS "Verified ${_actual} native tests are registered")
